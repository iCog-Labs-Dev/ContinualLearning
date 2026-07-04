import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu, relu_derivative


def compute_jacobians(weights, xs_eq):
    """Batch-averaged Jacobians used by the Schur–Fisher curvature."""
    jacobians = []
    for l in range(len(weights)):
        avg_mask = jnp.mean(relu_derivative(xs_eq[l]), axis=1)
        jacobians.append(weights[l] * avg_mask[None, :])
    return jacobians


def compute_jacobians_per_sample(weights, xs_eq):
    """Per-sample Jacobians used to compute per-sample Ã.

    For sample b and layer l:
        J^{(b)}_l[j, i] = W_l[j, i] · ReLU'(z_l^{(b)}[i]).

    Returns a list of `(B, d_out, d_in)` tensors. Uses only the per-sample
    ReLU mask; the linearised g_l = W · ReLU(·) then produces the standard
    Gauss-Newton local Jacobian in the natural-gradient surrogate. Fed into
    `compute_local_maps` via the `jacobians_per_sample=` kwarg; the batch-
    averaged Jacobians from `compute_jacobians` are still used to build the
    Schur curvature `Ḡ_bar`.
    """
    jacobians = []
    for l in range(len(weights)):
        mask = relu_derivative(xs_eq[l])                    # (d_in, B)
        # J^{(b)}[j, i] = W[j, i] · mask[i, b]
        jacobians.append(weights[l][None, :, :] * mask.T[:, None, :])  # (B, d_out, d_in)
    return jacobians


def compute_output_fisher(weights, xs_eq):
    """Categorical Fisher F_cat = mean_b [diag(p_b) − p_b p_bᵀ] at equilibrium.

    Used as the curvature at the output Schur base case (discriminative
    classification head). Replaces the Gaussian `diag(π_L)` everywhere the
    output edge enters the Schur recursion.
    """
    logits = weights[-1] @ relu(xs_eq[-2])
    probs = jax.nn.softmax(logits, axis=0)
    B = probs.shape[1]
    return jnp.diag(jnp.mean(probs, axis=1)) - (probs @ probs.T) / B


def compute_local_maps(
    precisions, lateral_pairs, jacobians, ridge=1e-4,
    output_fisher=None,
    jacobians_per_sample=None,
):
    """Schur–Fisher backward recursion for the per-layer local natural maps.

    `output_fisher` is the optional categorical Fisher matrix at the output
    edge. When provided, it replaces the Gaussian `diag(π_L)` in three
    places: the Schur base case `G_bar_L`, the output cross-term
    `PJ_{L-1} = F · J_{L-1}`, and the output's contribution to `H_{L-1}`
    via `J_{L-1}ᵀ · F · J_{L-1}`. Hidden layers keep the Gaussian diagonal
    machinery. If omitted, the output edge uses its diagonal precision.

    `lateral_pairs` is the low-rank lateral state: a list of
    `(alpha_eff, U)` tuples, one per hidden layer. The output edge has no
    lateral. Λ_l = α_l · U_l Uᵀ_l is materialised on demand to a dense
    (d, d) matrix inside `get_lateral`; the Schur recursion needs the dense
    form for H_l = Π + Λ + JᵀΠJ.

    `jacobians_per_sample`: optional list of
    `(B, d_out, d_in)` tensors. When provided, each `A_tildes[l]` is
    computed per-sample as `Ḡ_bar_{l+1}^{-1} · Π · J^{(b)}`. The Schur
    curvature (`build_H`, Schur update to `Ḡ`) still uses the batch-
    averaged `jacobians` — the Fisher metric is a data-average by design.
    When omitted, existing (batch-averaged) behaviour is preserved.
    """
    L = len(jacobians)
    # Shape check: lateral_pairs has one entry per hidden layer.
    expected_pairs = L - 1
    assert len(lateral_pairs) == expected_pairs, (
        f"compute_local_maps: lateral_pairs has {len(lateral_pairs)} "
        f"entries; expected {expected_pairs} (hidden layers only, output "
        f"edge excluded)."
    )
    for i, pair in enumerate(lateral_pairs):
        alpha, U = pair
        assert U.ndim == 2, f"lateral_pairs[{i}] U must be 2D, got shape {U.shape}"

    def get_lateral(layer_l):
        # layer_l ∈ {1..L} indexes xs[l] inside build_H. lateral_pairs[i] for
        # i = layer_l - 1 covers hidden layers xs[1..L-1]; xs[L] (output) gets
        # no lateral and returns None.
        i = layer_l - 1
        if 0 <= i < len(lateral_pairs):
            alpha, U = lateral_pairs[i]
            return alpha * (U @ U.T)
        return None

    def apply_above_curvature(l, X):
        """Left-multiply X by the curvature of the layer above (l+1).

        Output edge with `output_fisher` supplied: full matrix multiply
        by F_cat. Hidden layers use the diagonal precision
        `π[:, None] * X`.
        """
        if l == L - 1 and output_fisher is not None:
            return output_fisher @ X
        return precisions[l][:, None] * X

    def build_H(l):
        H = jnp.diag(precisions[l - 1])
        lat = get_lateral(l)
        if lat is not None:
            H = H + lat
        J_l = jacobians[l]
        H = H + J_l.T @ apply_above_curvature(l, J_l)
        return H

    if output_fisher is not None:
        G_bar_next = output_fisher
    else:
        G_bar_next = jnp.diag(precisions[L - 1])
    out_lateral = get_lateral(L)
    if out_lateral is not None:
        G_bar_next = G_bar_next + out_lateral

    use_per_sample = jacobians_per_sample is not None
    A_tildes = [None] * L
    for l in range(L - 1, -1, -1):
        # Batch-averaged PJ drives the Schur curvature recursion in either mode.
        PJ_batch = apply_above_curvature(l, jacobians[l])
        G_bar_next_reg = G_bar_next + ridge * jnp.eye(G_bar_next.shape[0])

        if use_per_sample:
            # Per-sample RHS: apply Π to each J^{(b)} then solve against the
            # shared batch-averaged Ḡ. `jnp.linalg.solve` broadcasts over the
            # leading batch dim (LU-decomposes Ḡ once, back-substitutes for B
            # right-hand sides).
            J_per = jacobians_per_sample[l]                       # (B, d_out, d_in)
            PJ_per = jax.vmap(lambda J: apply_above_curvature(l, J))(J_per)
            A_tildes[l] = jnp.linalg.solve(G_bar_next_reg, PJ_per)  # (B, d_out, d_in)
        else:
            A_tildes[l] = jnp.linalg.solve(G_bar_next_reg, PJ_batch)  # (d_out, d_in)

        if l > 0:
            # Schur update: uses the batch-averaged PJ (Fisher metric semantics).
            schur = PJ_batch.T @ jnp.linalg.solve(G_bar_next_reg, PJ_batch)
            G_bar_next = build_H(l) - schur

    return A_tildes


def compute_causal_gates(A_tildes, p=2.0, kappa=1e-3):
    """Row-normalised raw causal gate.

    Standard input: each `A_k` is a 2D `(d_out, d_in)` batch-averaged Ã.

        r_{ji} = |Ã_{ji}|^p / (Σ_{i'} |Ã_{ji'}|^p + κ)

    Per-sample input: each `A_k` is 3D `(B, d_out, d_in)` — the
    per-sample Ã stack from `compute_local_maps(..., jacobians_per_sample=)`.
    In that case `|Ã|^p` is averaged across the batch axis before the row
    normalisation:

        r_{ji} = mean_b(|Ã^{(b)}_{ji}|^p) / (Σ_{i'} mean_b(|Ã^{(b)}_{ji'}|^p) + κ)

    Output is always `(d_out, d_in)`. Properties: `r ∈ [0, 1)`;
    `Σ_i r_{ji} = 1 − κ/(Σ mean_b|Ã|^p + κ) ∈ [0, 1)`.
    """
    gates = []
    for A_k in A_tildes:
        abs_A_p = jnp.abs(A_k) ** p
        if abs_A_p.ndim == 3:
            abs_A_p = jnp.mean(abs_A_p, axis=0)  # average |Ã|^p across batch
        denom = jnp.sum(abs_A_p, axis=1, keepdims=True) + kappa
        gates.append(abs_A_p / denom)
    return gates


def compute_causal_gates_per_sample(A_tildes_per_sample, p=2.0, kappa=1e-3):
    """Per-sample row-normalised causal gate.

    Applies row-normalisation to each sample's own Ã^{(b)}, producing a
    stack of per-sample gates:

        r^{(b)}_{ji} = |Ã^{(b)}_{ji}|^p / (Σ_{i'} |Ã^{(b)}_{ji'}|^p + κ)

    Row-normalisation runs over parent index `i'`, per row `j`, **per
    sample b**. Batching is done by applying the Hebbian update per sample
    and averaging updates.

    Under ReLU + linear-W generative maps, the per-sample denominator
    `Σ_{i'} |Ã^{(b)}_{ji'}|^p = Σ_{i'} |C[j, i']|^p · mask^{(b)}[i']`
    sums only over parents that fired in sample b, so `r^{(b)}[j, i]`
    concentrates gate mass on the peak within the sample's firing
    subset. This breaks the batch-averaged `C · mask` factorisation
    that the batch-averaged gate summary collapses to.

    Input: list of `(B, d_out, d_in)` per-layer per-sample Ã tensors.
    Output: list of `(B, d_out, d_in)` per-layer per-sample gates in
    `[0, 1)`, with per-row row-sum ≈ `1 − κ / denom^{(b)}[j]`.

    Called by `training.train_step` for the per-sample Hebbian update.
    Diagnostics can still call `compute_causal_gates` on the
    same input for a 2D summary of the gate structure.
    """
    gates = []
    for A_k in A_tildes_per_sample:
        assert A_k.ndim == 3, (
            f"compute_causal_gates_per_sample expects (B, d_out, d_in) per "
            f"layer; got shape {A_k.shape}"
        )
        abs_A_p = jnp.abs(A_k) ** p                                # (B, d_out, d_in)
        denom = jnp.sum(abs_A_p, axis=2, keepdims=True) + kappa    # (B, d_out, 1)
        gates.append(abs_A_p / denom)                              # (B, d_out, d_in)
    return gates
