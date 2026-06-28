"""Learned low-rank PSD hidden lateral precision helpers.

Implements:
- `ramp_schedule`: warmup + linear ramp for the lateral force (Python-side).
- `lateral_force`: O(d·r·B) low-rank matvec for `Λ z` during inference.
- `materialize_lateral`: O(d²·r) materialization used once per Schur call.
- `lateral_loss_one_layer` / `total_lateral_loss`: covariance-to-precision
  objective with the matrix-determinant lemma identity.
- `adam_update_lateral`: hand-rolled Adam with global-norm gradient clipping.
  When `lr_scale == 0` the entire update is a no-op — `m`, `v`, AND step
  are all preserved so the next non-zero-scale update gets the correct
  bias-correction divisor.
- `apply_spectral_cap`: rescales `U` so `softplus(ρ) · λ_max(Uᵀ U) ≤ cap`.
  Uses the unramped lateral strength so `U` cannot drift unbounded while the
  force schedule is still below full strength.
- `update_cov_ema`: β·C_ema + (1−β)·(1/B) z @ zᵀ with stop_gradient on z.

These helpers keep the lateral precision energy-based and positive
semidefinite while avoiding dense lateral matvecs during inference.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg


# Schedule / parametrisation helpers

def ramp_schedule(epoch_idx, warmup_epochs, ramp_epochs):
    """0 during warmup, linear 0→1 over `ramp_epochs`, 1 thereafter.

    Pure-Python; called once per epoch in `method.train_task` before the
    JIT-compiled `train_step`. Returns a Python float.
    """
    if epoch_idx < warmup_epochs:
        return 0.0
    span = epoch_idx - warmup_epochs + 1
    if span >= ramp_epochs:
        return 1.0
    return float(span) / float(ramp_epochs)


def inv_softplus(x):
    """Inverse of softplus: log(exp(x) − 1) via expm1 for numerical safety."""
    return jnp.log(jnp.expm1(jnp.asarray(x)))

# Lateral force
def lateral_force(z, U, alpha):
    """Compute `Λ z` via low-rank matvec.

    Λ = α · U Uᵀ, so Λ z = α · U (Uᵀ z). Cost O(d·r·B), not O(d²·B).

    z shape: (d, batch).  U shape: (d, r).  alpha: scalar.
    Returns: (d, batch).
    """
    return alpha * (U @ (U.T @ z))


def materialize_lateral(U, alpha):
    """Materialise Λ = α · U Uᵀ as a (d, d) matrix.

    Used inside `do_influence.compute_local_maps` where the Schur recursion
    requires the dense form for the H_l = Π + Λ + JᵀΠJ matrix.
    """
    return alpha * (U @ U.T)


# Diffusion clarity helpers.

def diffusion_clarity_kernel(U, alpha, t):
    """Heat-kernel diffusion on the lateral graph at time t.

    For one hidden layer's lateral matrix Λ = α · U Uᵀ, build the
    unnormalised graph Laplacian L = D − W with W = |Λ| (diagonal of
    |Λ| included per VLCP §4) and D = diag(rowsum(W)). Return the heat
    kernel K_t = exp(−t · L) via `jax.scipy.linalg.expm`.

    `K_t[v, u]` measures the diffused influence from neuron u to neuron
    v through all paths in the layer. Used to identify direct edges
    that are reproducible by multi-hop indirect paths.

    Pure function; differentiable through both `expm` and the |Λ|
    construction.
    """
    Lam = materialize_lateral(U, alpha)
    W = jnp.abs(Lam)
    D = jnp.diag(jnp.sum(W, axis=1))
    L = D - W
    return jax.scipy.linalg.expm(-t * L)


def clarity_gap(U, alpha, t):
    """Per-edge clarity gap K_t − |Λ|, plus supporting matrices.

    Returns the tuple `(raw_gap, K_t, abs_Lam)` where:
        K_t     = exp(−t · L)            (heat kernel)
        abs_Lam = |α · U Uᵀ|             (direct-edge magnitudes)
        raw_gap = K_t − abs_Lam          (signed; positive = indirect > direct)

    The VLCP penalty integrand is `(raw_gap − ε)_+` summed over
    off-diagonal entries; callers apply ε and the u ≠ v mask.
    """
    K_t = diffusion_clarity_kernel(U, alpha, t)
    abs_Lam = jnp.abs(materialize_lateral(U, alpha))
    return K_t - abs_Lam, K_t, abs_Lam


def lateral_clarity_loss(U, alpha_eff, t, eps):
    """VLCP diffusion clarity penalty for one hidden layer (scalar).

        L_diff = Σ_{u≠v} ( K_t[v,u] − |Λ[v,u]| − ε )_+

    with Λ = α_eff · U Uᵀ (i.e. ramp-multiplied, so the term self-gates
    to zero during lateral warmup: at `ramp = 0` ⇒ `alpha_eff = 0` ⇒
    `|Λ| = 0` ⇒ `L = 0` ⇒ `K_t = I` ⇒ off-diagonal penalty = 0).

    The `u ≠ v` restriction in VLCP §4 is enforced by masking the
    diagonal of the relu output; the diagonal of `|Λ|` IS included in
    `W` when forming the Laplacian (per writeup), which is handled
    inside `diffusion_clarity_kernel`.

    Returns the scalar to be multiplied by `λ_d` in
    `lateral_loss_one_layer`. Differentiable through `expm` and the
    `|Λ|` construction.
    """
    raw_gap, _K_t, _abs_Lam = clarity_gap(U, alpha_eff, t)
    gap = raw_gap - eps
    d = U.shape[0]
    offdiag = 1.0 - jnp.eye(d)
    return jnp.sum(jax.nn.relu(gap) * offdiag)


# Lateral learning objective
def lateral_loss_one_layer(
    U,
    rho,
    C_ema,
    ramp,
    beta_logdet,
    eps_lat,
    lambda_fro,
    lambda_U,
    lambda_d=0.0,
    clarity_t=1.0,
    clarity_eps=1e-4,
):
    """Covariance-to-precision objective for one hidden layer.

    L_lat = (1/2) tr(Λ C_ema)
            − β_logdet · log det(Λ + ε_lat · I)
            + λ_fro · ||Λ||_F²
            + λ_U · ||U||_F²
            + λ_d · Σ_{u≠v} ( K_t[v,u] − |Λ[v,u]| − ε )_+   (VLCP §4)

    with Λ = ramp · softplus(ρ) · U Uᵀ. Uses the matrix-determinant lemma
    `log det(εI + α U Uᵀ) = d·log(ε) + log det(I_r + (α/ε) Uᵀ U)`; the
    `d·log(ε)` constant is dropped from the loss (no gradient).

    `C_ema` is expected to be stop_gradient-wrapped by the caller.
    Returns a scalar. With `lambda_d = 0`, the clarity term has no
    semantic effect and preserves the pre-Phase-6a lateral objective.
    """
    raw_alpha = jax.nn.softplus(rho)
    alpha_eff = ramp * raw_alpha
    r = U.shape[1]

    # (1/2) tr(Λ C) = (1/2) α_eff · tr(Uᵀ C U)
    CU = C_ema @ U                              # (d, r)
    trace_term = 0.5 * alpha_eff * jnp.sum(U * CU)

    # log det(εI + α_eff U Uᵀ), dropping the constant d·log(ε)
    UtU = U.T @ U                               # (r, r)
    M = jnp.eye(r) + (alpha_eff / eps_lat) * UtU
    _sign, logdet = jnp.linalg.slogdet(M)

    # ||Λ||_F² = α_eff² · ||Uᵀ U||_F²
    frob_term = lambda_fro * (alpha_eff ** 2) * jnp.sum(UtU ** 2)

    # Direct U L2 decay
    U_l2_term = lambda_U * jnp.sum(U ** 2)

    # VLCP §4 diffusion clarity penalty. Self-gated to zero during
    # warmup via alpha_eff = ramp · raw_alpha.
    clarity_term = lambda_d * lateral_clarity_loss(
        U, alpha_eff, clarity_t, clarity_eps
    )

    return (
        trace_term - beta_logdet * logdet + frob_term + U_l2_term + clarity_term
    )


def total_lateral_loss(
    Us,
    rhos,
    C_emas,
    ramp,
    beta_logdet,
    eps_lat,
    lambda_fro,
    lambda_U,
    lambda_d=0.0,
    clarity_t=1.0,
    clarity_eps=1e-4,
):
    """Sum of per-layer lateral losses across hidden layers.

    `Us`, `rhos`, `C_emas` are parallel lists (length = num_hidden).
    Returns a scalar suitable for `jax.grad`. Defaults for the VLCP
    §4 clarity kwargs reproduce the pre-Phase-6a behaviour.
    """
    total = jnp.array(0.0)
    for U, rho, C in zip(Us, rhos, C_emas):
        total = total + lateral_loss_one_layer(
            U, rho, C, ramp, beta_logdet, eps_lat, lambda_fro, lambda_U,
            lambda_d, clarity_t, clarity_eps,
        )
    return total


# Hand-rolled Adam
def adam_init_per_layer(U, rho):
    """Initialise Adam state matching shapes of `(U, rho)`."""
    return {
        "m_U": jnp.zeros_like(U),
        "v_U": jnp.zeros_like(U),
        "m_rho": jnp.zeros_like(rho),
        "v_rho": jnp.zeros_like(rho),
    }


def _global_grad_norm(grad_Us, grad_rhos):
    sq_sum = jnp.array(0.0)
    for g in grad_Us:
        sq_sum = sq_sum + jnp.sum(g ** 2)
    for g in grad_rhos:
        sq_sum = sq_sum + jnp.sum(g ** 2)
    return jnp.sqrt(sq_sum + 1e-12)


def adam_update_lateral(
    Us,
    rhos,
    grad_Us,
    grad_rhos,
    adam_states,
    step_counter,
    lr_lat,
    lr_scale,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    clip_norm=1.0,
):
    """One Adam step for all lateral layers with global-norm clipping.

    `lr_scale` is a float ∈ {0.0, 1.0} (set by `method.train_task` from the
    warmup gate). When `lr_scale == 0.0` the update is a no-op:
        - `U`, `ρ` unchanged
        - `m`, `v` unchanged (kept at previous values)
        - `step_counter` not incremented (so the first non-zero-scale update
          sees the correct bias-correction divisor `1 − β1^1`).

    Returns `(new_Us, new_rhos, new_adam_states, new_step_counter)`.
    """
    # Global-norm clipping on the lateral grad ensemble
    g_norm = _global_grad_norm(grad_Us, grad_rhos)
    clip_scale = jnp.minimum(1.0, clip_norm / (g_norm + 1e-12))

    new_step = step_counter + lr_scale

    new_Us = []
    new_rhos = []
    new_states = []

    for U, rho, gU, gR, st in zip(Us, rhos, grad_Us, grad_rhos, adam_states):
        gU = gU * clip_scale
        gR = gR * clip_scale

        # Candidate new moments (computed even in warmup; gated below)
        m_U_cand = beta1 * st["m_U"] + (1.0 - beta1) * gU
        v_U_cand = beta2 * st["v_U"] + (1.0 - beta2) * (gU ** 2)
        m_rho_cand = beta1 * st["m_rho"] + (1.0 - beta1) * gR
        v_rho_cand = beta2 * st["v_rho"] + (1.0 - beta2) * (gR ** 2)

        # Bias correction at the candidate new step count.
        bc1 = jnp.maximum(1.0 - beta1 ** new_step, 1e-12)
        bc2 = jnp.maximum(1.0 - beta2 ** new_step, 1e-12)

        m_U_hat = m_U_cand / bc1
        v_U_hat = v_U_cand / bc2
        m_rho_hat = m_rho_cand / bc1
        v_rho_hat = v_rho_cand / bc2

        U_step = lr_lat * m_U_hat / (jnp.sqrt(v_U_hat) + eps)
        rho_step = lr_lat * m_rho_hat / (jnp.sqrt(v_rho_hat) + eps)

        # Gate everything by lr_scale ∈ {0, 1}: convex combo with previous
        U_new = U - lr_scale * U_step
        rho_new = rho - lr_scale * rho_step

        m_U_final = lr_scale * m_U_cand + (1.0 - lr_scale) * st["m_U"]
        v_U_final = lr_scale * v_U_cand + (1.0 - lr_scale) * st["v_U"]
        m_rho_final = lr_scale * m_rho_cand + (1.0 - lr_scale) * st["m_rho"]
        v_rho_final = lr_scale * v_rho_cand + (1.0 - lr_scale) * st["v_rho"]

        new_Us.append(U_new)
        new_rhos.append(rho_new)
        new_states.append(
            {
                "m_U": m_U_final,
                "v_U": v_U_final,
                "m_rho": m_rho_final,
                "v_rho": v_rho_final,
            }
        )

    return new_Us, new_rhos, new_states, new_step


# Spectral cap and covariance EMA
def apply_spectral_cap(U, rho, lambda_max_cap):
    """Rescale `U` so that `softplus(ρ) · λ_max(Uᵀ U) ≤ lambda_max_cap`.

    The cap is applied to the unramped lateral strength, so the maximum
    possible force remains bounded even while the training schedule is still
    below full strength.

    Only `U` is rescaled — `ρ` is left unchanged so the user-visible α
    parameter stays interpretable.
    """
    raw_alpha = jax.nn.softplus(rho)
    UtU = U.T @ U
    eigs = jnp.linalg.eigvalsh(UtU)
    lam_max_raw = raw_alpha * jnp.max(eigs)
    scale = jnp.sqrt(jnp.minimum(1.0, lambda_max_cap / (lam_max_raw + 1e-12)))
    return U * scale


def update_cov_ema(C_ema, z_eq, beta_cov):
    """C_ema ← β · C_ema + (1−β) · (1/B) · z @ zᵀ.

    `z_eq` shape: (d, batch). Uncentered second moment, matching the
    zero-mean Gaussian prior form `(1/2) zᵀ Λ z` we are aligning Λ to.

    `stop_gradient` on `z_eq` so lateral training doesn't backprop into the
    PC inference / weight machinery.
    """
    z_eq = jax.lax.stop_gradient(z_eq)
    B = z_eq.shape[1]
    C_batch = (z_eq @ z_eq.T) / B
    return beta_cov * C_ema + (1.0 - beta_cov) * C_batch
