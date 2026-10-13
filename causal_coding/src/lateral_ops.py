"""Unified lateral operator module — structured lateral matrix experiment.

Defines the data types and the four seam functions every caller uses:

    build_lateral_ops  – params → list of ops (one per hidden layer)
    apply_lateral      – op, z → lateral force subtracted during inference
    dense_lateral      – op → dense (d, d) matrix for the Schur step
    guard              – enforce Gershgorin row-budget on (s, b)

Also provides:

    init_structured    – initialise the four structured param lists
    struct_update      – one local learning-rule step (not a gradient)
    StructLateral      – NamedTuple: (s, b) already ramp-scaled
    BothLateral        – NamedTuple: (lowrank, structured)

Design invariants
-----------------
* Pure functions, no global state.
* Default lateral_mode = "lowrank" produces bit-identical output to the
  pre-patch code (same arithmetic, same Python objects).
* New param keys exist only when mode in ("structured", "both").
* mode strings and struct_learning are jax.jit STATIC arguments.
* Strengths / lr / targets are traced floats.
* guard() is always applied when structured strengths can change;
  the firing fraction is stored in params["struct_guard_frac"] for logging.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class StructLateral(NamedTuple):
    """Structured lateral operator for one hidden layer.

    Attributes
    ----------
    s : Array  (d,)
        Per-unit self-excitation strengths, already ramp-scaled.
    b : Array  (G, K, K)
        Within-group inhibition matrix, already ramp-scaled.
        Symmetric, non-negative, zero diagonal.  G = d // K.
    """
    s: jax.Array   # (d,)
    b: jax.Array   # (G, K, K)


class BothLateral(NamedTuple):
    """Combined low-rank + structured operator for one hidden layer.

    Attributes
    ----------
    lowrank : tuple (alpha_eff, U)
        Exactly the 2-tuple used today; alpha_eff is ramp-scaled.
    structured : StructLateral
        Already ramp-scaled.
    """
    lowrank:    tuple          # (alpha_eff, U)
    structured: StructLateral


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_structured(hidden_dims, K, alpha_e=0.13, inhib_total=0.25, rho=0.2):
    """Return the four structured param lists (one entry per hidden layer).

    Parameters
    ----------
    hidden_dims : list[int]
        Width of each hidden layer.  Every d must be divisible by K.
    K : int
        Block (group) size.
    alpha_e : float
        Initial per-unit self-excitation (s_i at init).
    inhib_total : float
        Total inhibitory budget per unit: sum_j b_ij = inhib_total.
        Split uniformly over K-1 off-diagonal entries: b_ij = inhib_total/(K-1).
    rho : float
        Initial EMA activity prior (p_ema=rho, q_ema=rho^2 → stationary start).

    Returns
    -------
    struct_s     : list of (d,) arrays
    struct_b     : list of (G, K, K) arrays
    struct_p_ema : list of (d,) arrays  — running active fraction
    struct_q_ema : list of (G, K, K) arrays — running co-activity
    """
    struct_s, struct_b, struct_p_ema, struct_q_ema = [], [], [], []
    for d in hidden_dims:
        assert d % K == 0, (
            f"Hidden dim {d} is not divisible by group size K={K}."
        )
        G = d // K

        s = jnp.full((d,), alpha_e, dtype=jnp.float32)

        b_val = inhib_total / (K - 1) if K > 1 else 0.0
        block = b_val * (jnp.ones((K, K), dtype=jnp.float32)
                         - jnp.eye(K, dtype=jnp.float32))
        b = jnp.stack([block] * G)   # (G, K, K)

        p_ema = jnp.full((d,), rho, dtype=jnp.float32)
        q_ema = jnp.full((G, K, K), rho ** 2, dtype=jnp.float32)

        struct_s.append(s)
        struct_b.append(b)
        struct_p_ema.append(p_ema)
        struct_q_ema.append(q_ema)

    return struct_s, struct_b, struct_p_ema, struct_q_ema


# ---------------------------------------------------------------------------
# Building the operator list
# ---------------------------------------------------------------------------

def build_lateral_ops(params, mode, force_scale):
    """Build the per-layer lateral operator list from params.

    Parameters
    ----------
    params : dict
        Model parameter dict.
    mode : str  (STATIC jit argument)
        "lowrank"    – plain (alpha_eff, U) tuple, bit-identical to today.
        "structured" – StructLateral only.
        "both"       – BothLateral.
    force_scale : float
        Lateral force ramp in [0, 1].

    Returns
    -------
    list of ops, one per hidden layer.
    """
    if mode == "lowrank":
        raw_alphas = [jax.nn.softplus(rho)
                      for rho in params["lateral_log_alpha"]]
        eff_alphas = [force_scale * a for a in raw_alphas]
        return list(zip(eff_alphas, params["lateral_U"]))

    if mode == "structured":
        return [
            StructLateral(force_scale * s, force_scale * b)
            for s, b in zip(params["struct_s"], params["struct_b"])
        ]

    # mode == "both"
    raw_alphas = [jax.nn.softplus(rho)
                  for rho in params["lateral_log_alpha"]]
    ops = []
    for rho_raw, U, s, b in zip(
        raw_alphas,
        params["lateral_U"],
        params["struct_s"],
        params["struct_b"],
    ):
        ops.append(BothLateral(
            lowrank=(force_scale * rho_raw, U),
            structured=StructLateral(force_scale * s, force_scale * b),
        ))
    return ops


# ---------------------------------------------------------------------------
# Seam: apply_lateral
# ---------------------------------------------------------------------------

def apply_lateral(op, z):
    """Compute the lateral force that _infer_step subtracts from the update.

    Update rule:  z <- z + lr_z * (top_down - weighted_self - lateral)
    This function returns `lateral`.

    Sign convention (V = B - diag(s)):
        lateral_i = sum_j B_ij*relu(z_j)  -  s_i*relu(z_i)

    Subtracting from the update gives:
        +s_i*relu(z_i)          (self-excitation)
        -sum_j B_ij*relu(z_j)   (within-group inhibition)

    Parameters
    ----------
    op : StructLateral | BothLateral | plain 2-tuple (alpha_eff, U)
    z  : (d, B) array

    Returns
    -------
    (d, B) array
    """
    # StructLateral check before plain tuple (both are iterable).
    if isinstance(op, StructLateral):
        G, K, _ = op.b.shape
        phi = relu(z)                                          # (d, B)
        phi_g = phi.reshape(G, K, -1)                         # (G, K, B)
        Bphi_g = jnp.einsum("gij,gjb->gib", op.b, phi_g)     # (G, K, B)
        Bphi = Bphi_g.reshape(z.shape)                        # (d, B)
        return Bphi - op.s[:, None] * phi                     # V·relu(z)

    if isinstance(op, BothLateral):
        return apply_lateral(op.lowrank, z) + apply_lateral(op.structured, z)

    # Plain 2-tuple: (alpha_eff, U) — identical to today.
    alpha, U = op
    return alpha * (U @ (U.T @ z))


# ---------------------------------------------------------------------------
# Seam: dense_lateral
# ---------------------------------------------------------------------------

def dense_lateral(op, mask=None, include_structured=False):
    """Materialise the lateral contribution as a dense (d, d) matrix.

    Used by compute_local_maps to build H_l = Pi + Lambda + J^T Pi J.

    Parameters
    ----------
    op : StructLateral | BothLateral | plain 2-tuple
    mask : (d,) array or None
        Batch-averaged relu' of the layer.  When given, the structured matrix
        is symmetrised as D^{1/2} V D^{1/2} (Stage F, mask-weighted form).
    include_structured : bool  (STATIC jit argument)
        When False (default), structured lateral is NOT included in the Schur
        matrix — conservative safe default.  Set True in struct_metrics and
        for Stage F experiments.

    Returns
    -------
    (d, d) array or None.
    """
    if isinstance(op, StructLateral):
        if not include_structured:
            return None
        return _dense_struct(op, mask)

    if isinstance(op, BothLateral):
        lr_dense = _dense_lowrank(*op.lowrank)
        if include_structured:
            return lr_dense + _dense_struct(op.structured, mask)
        return lr_dense

    # Plain 2-tuple.
    return _dense_lowrank(*op)


def _dense_lowrank(alpha, U):
    return alpha * (U @ U.T)


def _dense_struct(op, mask):
    """Assemble the full dense V matrix from a StructLateral.

    V[i,j] = -s_i      if i == j
            = B_g[k,l]  if same block g, k=i%K, l=j%K, i!=j
            = 0          otherwise
    """
    G, K, _ = op.b.shape
    d = G * K
    eye_G = jnp.eye(G, dtype=op.b.dtype)
    # (G,K,K) x (G,G) -> (G,K,G,K) -> (d,d) block diagonal B
    B_dense = jnp.einsum("gij,gh->gihj", op.b, eye_G).reshape(d, d)
    V = B_dense - jnp.diag(op.s)
    if mask is not None:
        sqrt_mask = jnp.sqrt(jnp.maximum(mask, 0.0))
        V = sqrt_mask[:, None] * V * sqrt_mask[None, :]
    return V


# ---------------------------------------------------------------------------
# Guard: Gershgorin row-budget
# ---------------------------------------------------------------------------

def guard(s, b, pi, c=0.5):
    """Enforce Gershgorin row-budget so diag(Pi) + V is positive definite.

    Row budget:  s_i + sum_j b_ij  <=  c * Pi_i

    Guarantee:  lambda_min(diag(Pi) + V)  >=  (1 - c) * min(Pi)

    Parameters
    ----------
    s  : (d,)     self-excitation (>= 0)
    b  : (G,K,K)  inhibition (>= 0, symmetric, zero diagonal)
    pi : (d,)     current precision diagonal
    c  : float    budget fraction in (0, 1)

    Returns
    -------
    s_new, b_new, fraction_scaled
        fraction_scaled: mean fraction of rows that were clipped, in [0,1].
    """
    G, K, _ = b.shape
    row_sum_b = b.sum(axis=-1).reshape(-1)        # (d,)
    r = s + row_sum_b                              # total row budget (d,)

    kappa = jnp.minimum(1.0, c * pi / (r + 1e-12))   # (d,) in (0, 1]

    s_new = s * kappa

    # Symmetric scaling of b: scale pair (i,j) by min(kappa_i, kappa_j).
    kg = kappa.reshape(G, K)                                      # (G, K)
    kb = jnp.minimum(kg[:, :, None], kg[:, None, :])             # (G, K, K)
    b_new = b * kb

    fraction_scaled = jnp.mean((kappa < 1.0).astype(jnp.float32))
    return s_new, b_new, fraction_scaled


# ---------------------------------------------------------------------------
# Local learning rule
# ---------------------------------------------------------------------------

def struct_update(
    s, b, p_ema, q_ema,
    z_eq, pi_new,
    lr_s, lr_b,
    rho, beta_stat, guard_c,
    struct_learning,   # "fixed" | "scalar" | "perunit"  (STATIC)
    lr_scale,          # 0.0 = frozen; 1.0 = active  (traced)
):
    """One structured lateral learning-rule step.

    Uses binary activity a_i = 1[z_i > 0] so rho is an active fraction.

    Equations
    ---------
    p_i  = mean_batch a_i
    q_ij = mean_batch a_i * a_j   (within block only)
    EMA both with beta_stat.

    Delta_s_i  = lr_s * (rho - p_ema_i)      too quiet -> more excitation
    Delta_b_ij = lr_b * (q_ema_ij - rho^2)   co-active -> more inhibition

    Fixed point: p_i = rho, q_ij = rho^2 (independent units at rate rho).

    Tie rules (struct_learning, static)
    ------------------------------------
    "fixed"   : statistics updated; Delta_s = 0, Delta_b = 0.
    "scalar"  : Delta_s = mean over units (broadcast).
                Delta_b = mean over off-diagonal entries (broadcast).
                guard uses min(Pi) so uniformity survives.
    "perunit" : each unit / pair moves independently.

    lr_scale gates the whole parameter update (0 = frozen).

    Returns
    -------
    s_new, b_new, p_ema_new, q_ema_new, guard_frac
    """
    G, K, _ = b.shape
    B_batch = z_eq.shape[1]

    # Binary activity: (d, B_batch)
    a = (z_eq > 0.0).astype(jnp.float32)

    # Active fraction per unit.
    p_batch = jnp.mean(a, axis=1)                          # (d,)

    # Co-activity within blocks: (G, K, K)
    a_g = a.reshape(G, K, B_batch)
    q_batch = jnp.einsum("gib,gjb->gij", a_g, a_g) / B_batch

    # EMA update.
    p_ema_new = beta_stat * p_ema + (1.0 - beta_stat) * p_batch
    q_ema_new = beta_stat * q_ema + (1.0 - beta_stat) * q_batch

    # Off-diagonal mask for b updates.
    eye_K = jnp.eye(K, dtype=b.dtype)
    off_mask = 1.0 - eye_K[None, :, :]   # (1, K, K) broadcast to (G, K, K)

    if struct_learning == "fixed":
        # Statistics only, no parameter change.
        s_new, b_new, guard_frac = guard(s, b, pi_new, guard_c)
        return s_new, b_new, p_ema_new, q_ema_new, guard_frac

    # Raw deltas.
    delta_s = lr_s * (rho - p_ema_new)                    # (d,)
    delta_b = lr_b * (q_ema_new - rho ** 2) * off_mask    # (G, K, K)

    if struct_learning == "scalar":
        delta_s = jnp.full_like(s, jnp.mean(delta_s))
        n_off = jnp.sum(off_mask)
        delta_b = jnp.full_like(b,
                                 jnp.sum(delta_b) / (n_off + 1e-12)) * off_mask
        pi_for_guard = jnp.full_like(pi_new, jnp.min(pi_new))
    else:
        # "perunit"
        pi_for_guard = pi_new

    # Gate by lr_scale (0 = frozen, 1 = active).
    delta_s = lr_scale * delta_s
    delta_b = lr_scale * delta_b

    # Apply and project.
    s_cand = jnp.maximum(0.0, s + delta_s)
    b_cand = jnp.maximum(0.0, b + delta_b) * off_mask
    # Symmetrise (fp drift guard).
    b_cand = 0.5 * (b_cand + b_cand.transpose(0, 2, 1))

    # Gershgorin guard.
    s_new, b_new, guard_frac = guard(s_cand, b_cand, pi_for_guard, guard_c)

    return s_new, b_new, p_ema_new, q_ema_new, guard_frac
