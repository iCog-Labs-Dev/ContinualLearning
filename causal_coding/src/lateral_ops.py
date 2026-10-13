"""Unified lateral operator module: structured lateral matrix experiment.

Seam functions every caller uses:

    build_lateral_ops  params -> list of ops (one per hidden layer)
    apply_lateral      op, z -> lateral force subtracted during inference
    dense_lateral      op -> dense (d, d) matrix for the Schur step (or None)
    guard              enforce Gershgorin row-budget on (s, b)

Also: init_structured, struct_update (local rule, not a gradient),
StructLateral, BothLateral.

Design invariants
-----------------
* Pure functions, no global state.
* mode == "lowrank" returns plain (alpha_eff, U) tuples and the same
  arithmetic as the pre-patch code, so old runs reproduce bit-for-bit.
* New param keys exist only when mode in ("structured", "both").
* mode, struct_learning and include_structured are jax.jit STATIC arguments.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu

VALID_MODES = ("lowrank", "structured", "both")
VALID_LEARNING = ("fixed", "scalar", "perunit")


class StructLateral(NamedTuple):
    """Structured lateral operator for one hidden layer (ramp-scaled).

    s : (d,)      self-excitation, >= 0
    b : (G, K, K) within-group inhibition, symmetric, >= 0, zero diagonal
    """
    s: jax.Array
    b: jax.Array


class BothLateral(NamedTuple):
    """Low-rank (alpha_eff, U) plus structured operator for one layer."""
    lowrank: tuple
    structured: StructLateral


def init_structured(hidden_dims, K, alpha_e=0.13, inhib_total=0.25, rho=0.2):
    """Return (struct_s, struct_b, struct_p_ema, struct_q_ema) lists."""
    struct_s, struct_b, struct_p_ema, struct_q_ema = [], [], [], []
    for d in hidden_dims:
        assert d % K == 0, f"Hidden dim {d} is not divisible by group size K={K}."
        G = d // K
        s = jnp.full((d,), alpha_e, dtype=jnp.float32)
        b_val = inhib_total / (K - 1) if K > 1 else 0.0
        block = b_val * (jnp.ones((K, K), dtype=jnp.float32)
                         - jnp.eye(K, dtype=jnp.float32))
        b = jnp.stack([block] * G)
        struct_s.append(s)
        struct_b.append(b)
        struct_p_ema.append(jnp.full((d,), rho, dtype=jnp.float32))
        struct_q_ema.append(jnp.full((G, K, K), rho ** 2, dtype=jnp.float32))
    return struct_s, struct_b, struct_p_ema, struct_q_ema


def build_lateral_ops(params, mode, force_scale):
    """Build the per-layer lateral operator list. `mode` is static."""
    if mode not in VALID_MODES:
        raise ValueError(f"lateral_mode must be one of {VALID_MODES}, got {mode!r}")

    if mode == "lowrank":
        raw_alphas = [jax.nn.softplus(rho) for rho in params["lateral_log_alpha"]]
        eff_alphas = [force_scale * a for a in raw_alphas]
        return list(zip(eff_alphas, params["lateral_U"]))

    if mode == "structured":
        return [
            StructLateral(force_scale * s, force_scale * b)
            for s, b in zip(params["struct_s"], params["struct_b"])
        ]

    raw_alphas = [jax.nn.softplus(rho) for rho in params["lateral_log_alpha"]]
    return [
        BothLateral(
            lowrank=(force_scale * a, U),
            structured=StructLateral(force_scale * s, force_scale * b),
        )
        for a, U, s, b in zip(
            raw_alphas, params["lateral_U"], params["struct_s"], params["struct_b"]
        )
    ]


def apply_lateral(op, z):
    """Lateral force that _infer_step subtracts. z is (d, B) -> (d, B).

    With V = B - diag(s) the force is V.relu(z); subtracting it gives
    +s_i relu(z_i) (self-excitation) and -sum_j B_ij relu(z_j) (inhibition).
    """
    # NamedTuples are tuples, so check them before the plain 2-tuple case.
    if isinstance(op, StructLateral):
        G, K, _ = op.b.shape
        phi = relu(z)
        Bphi = jnp.einsum("gij,gjb->gib", op.b, phi.reshape(G, K, -1)).reshape(z.shape)
        return Bphi - op.s[:, None] * phi
    if isinstance(op, BothLateral):
        return apply_lateral(op.lowrank, z) + apply_lateral(op.structured, z)
    alpha, U = op
    return alpha * (U @ (U.T @ z))


def dense_lateral(op, mask=None, include_structured=False):
    """Dense (d, d) lateral matrix for the Schur step, or None.

    include_structured is STATIC. With mask (batch-mean relu' of the layer)
    the structured part is symmetrised as D^{1/2} V D^{1/2}.
    """
    if isinstance(op, StructLateral):
        return _dense_struct(op, mask) if include_structured else None
    if isinstance(op, BothLateral):
        lr_dense = _dense_lowrank(*op.lowrank)
        if include_structured:
            return lr_dense + _dense_struct(op.structured, mask)
        return lr_dense
    return _dense_lowrank(*op)


def _dense_lowrank(alpha, U):
    return alpha * (U @ U.T)


def _dense_struct(op, mask):
    G, K, _ = op.b.shape
    d = G * K
    eye_G = jnp.eye(G, dtype=op.b.dtype)
    B_dense = jnp.einsum("gij,gh->gihj", op.b, eye_G).reshape(d, d)
    V = B_dense - jnp.diag(op.s)
    if mask is not None:
        sq = jnp.sqrt(jnp.maximum(mask, 0.0))
        V = sq[:, None] * V * sq[None, :]
    return V


def guard(s, b, pi, c=0.5):
    """Enforce row budget s_i + sum_j b_ij <= c * Pi_i (Gershgorin).

    Guarantee: lambda_min(diag(Pi) + V) >= (1 - c) * min(Pi).
    Returns (s_new, b_new, fraction_of_rows_scaled).
    """
    G, K, _ = b.shape
    r = s + b.sum(axis=-1).reshape(-1)
    kappa = jnp.minimum(1.0, c * pi / (r + 1e-12))
    kg = kappa.reshape(G, K)
    kb = jnp.minimum(kg[:, :, None], kg[:, None, :])   # keeps b symmetric
    frac = jnp.mean((kappa < 1.0).astype(jnp.float32))
    return s * kappa, b * kb, frac


def struct_update(
    s, b, p_ema, q_ema, z_eq, pi_new,
    lr_s, lr_b, rho, beta_stat, guard_c,
    struct_learning, lr_scale,
):
    """One local learning-rule step. struct_learning is STATIC.

    a = 1[z > 0];  p_i = mean_b a_i;  q_ij = mean_b a_i a_j (within block);
    both EMA'd.   Delta s_i = lr_s (rho - p_i);  Delta b_ij = lr_b (q_ij - rho^2).
    Returns (s_new, b_new, p_ema_new, q_ema_new, guard_frac).
    """
    if struct_learning not in VALID_LEARNING:
        raise ValueError(f"struct_learning must be one of {VALID_LEARNING}")

    G, K, _ = b.shape
    n_batch = z_eq.shape[1]

    a = (z_eq > 0.0).astype(jnp.float32)
    p_batch = jnp.mean(a, axis=1)
    a_g = a.reshape(G, K, n_batch)
    q_batch = jnp.einsum("gib,gjb->gij", a_g, a_g) / n_batch

    p_ema_new = beta_stat * p_ema + (1.0 - beta_stat) * p_batch
    q_ema_new = beta_stat * q_ema + (1.0 - beta_stat) * q_batch

    off_mask = 1.0 - jnp.eye(K, dtype=b.dtype)[None, :, :]

    if struct_learning == "fixed":
        s_new, b_new, frac = guard(s, b, pi_new, guard_c)
        return s_new, b_new, p_ema_new, q_ema_new, frac

    delta_s = lr_s * (rho - p_ema_new)
    delta_b = lr_b * (q_ema_new - rho ** 2) * off_mask

    if struct_learning == "scalar":
        delta_s = jnp.full_like(s, jnp.mean(delta_s))
        # Mean over ALL off-diagonal entries of ALL blocks: G * K * (K-1).
        n_off_total = G * jnp.sum(off_mask)
        delta_b = jnp.full_like(b, jnp.sum(delta_b) / (n_off_total + 1e-12)) * off_mask
        pi_for_guard = jnp.full_like(pi_new, jnp.min(pi_new))
    else:
        pi_for_guard = pi_new

    delta_s = lr_scale * delta_s
    delta_b = lr_scale * delta_b

    s_cand = jnp.maximum(0.0, s + delta_s)
    b_cand = jnp.maximum(0.0, b + delta_b) * off_mask
    b_cand = 0.5 * (b_cand + b_cand.transpose(0, 2, 1))

    s_new, b_new, frac = guard(s_cand, b_cand, pi_for_guard, guard_c)
    return s_new, b_new, p_ema_new, q_ema_new, frac
