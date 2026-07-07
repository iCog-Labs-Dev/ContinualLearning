"""VLCP vertical soft-pruning helpers.

Implements soft per-synapse gates and regression-based importance scores.
The gates modulate weights but do not physically delete them.
"""

import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu
from causal_coding.src.inference import infer_with_trajectory
from causal_coding.src.vlcp_regression import compute_dreg_for_pair


def logit(p):
    p = jnp.asarray(p)
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p / (1.0 - p))


def init_vertical_gate_logits(weights, initial_gate=0.99):
    init_value = logit(initial_gate)
    return [jnp.full_like(W, init_value) for W in weights]


def init_vertical_importance(weights):
    return [jnp.ones_like(W) for W in weights]


def vertical_adam_init(logits):
    return [{"m": jnp.zeros_like(h), "v": jnp.zeros_like(h)} for h in logits]


def vertical_gates(logits):
    return [jax.nn.sigmoid(h) for h in logits]


def effective_vertical_gates(logits, layer_scales):
    gates = vertical_gates(logits)
    return [
        1.0 - jnp.asarray(scale, dtype=G.dtype) + jnp.asarray(scale, dtype=G.dtype) * G
        for G, scale in zip(gates, layer_scales)
    ]


def apply_vertical_gates(weights, logits, layer_scales):
    gates_eff = effective_vertical_gates(logits, layer_scales)
    return [G * W for W, G in zip(weights, gates_eff)]


def _quantile_flat(x, q):
    flat = jnp.sort(jnp.ravel(x))
    idx = int(q * (flat.shape[0] - 1))
    return flat[idx]


def normalize_importance(d_raw, eps=1e-12):
    d_abs = jnp.abs(d_raw)
    p99 = _quantile_flat(d_abs, 0.99)
    denom = jnp.maximum(p99, eps)
    return jnp.clip(d_abs / denom, 0.0, 1.0)


def normalize_importance_list(d_raw_list, eps=1e-12):
    return [normalize_importance(d, eps=eps) for d in d_raw_list]


def vertical_gate_loss_components(
    logits,
    importance,
    alpha_g,
    lambda_match,
    lambda_vert,
    eps,
    layer_scales,
):
    """Return `(total, match, sparse)` for vertical gate losses."""
    if not logits:
        z = jnp.asarray(0.0)
        return z, z, z

    dtype = logits[0].dtype
    match = jnp.asarray(0.0, dtype=dtype)
    sparse = jnp.asarray(0.0, dtype=dtype)

    for h, d, scale in zip(logits, importance, layer_scales):
        G = jax.nn.sigmoid(h)
        d_stop = jax.lax.stop_gradient(d)
        target = jax.nn.sigmoid(alpha_g * d_stop)
        scale = jnp.asarray(scale, dtype=dtype)
        match = match + scale * lambda_match * jnp.mean((G - target) ** 2)
        sparse = sparse + scale * lambda_vert * jnp.mean(G / (d_stop + eps))

    return match + sparse, match, sparse


def _global_grad_norm(grads):
    sq_sum = jnp.asarray(0.0)
    for g in grads:
        sq_sum = sq_sum + jnp.sum(g ** 2)
    return jnp.sqrt(sq_sum + 1e-12)


def adam_update_vertical(
    logits,
    grads,
    adam_states,
    step_counter,
    lr_vert,
    lr_scale,
    layer_scales,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    clip_norm=1.0,
):
    """Adam update for gate logits with warmup and per-layer participation."""
    g_norm = _global_grad_norm(grads)
    clip_scale = jnp.minimum(1.0, clip_norm / (g_norm + 1e-12))
    new_step = step_counter + lr_scale

    new_logits = []
    new_states = []
    for h, g, st, layer_scale in zip(logits, grads, adam_states, layer_scales):
        g = g * clip_scale
        m_cand = beta1 * st["m"] + (1.0 - beta1) * g
        v_cand = beta2 * st["v"] + (1.0 - beta2) * (g ** 2)

        bc1 = jnp.maximum(1.0 - beta1 ** new_step, 1e-12)
        bc2 = jnp.maximum(1.0 - beta2 ** new_step, 1e-12)
        m_hat = m_cand / bc1
        v_hat = v_cand / bc2
        h_step = lr_vert * m_hat / (jnp.sqrt(v_hat) + eps)

        layer_lr_scale = lr_scale * jnp.where(jnp.asarray(layer_scale) > 0.0, 1.0, 0.0)
        h_new = h - layer_lr_scale * h_step
        m_new = layer_lr_scale * m_cand + (1.0 - layer_lr_scale) * st["m"]
        v_new = layer_lr_scale * v_cand + (1.0 - layer_lr_scale) * st["v"]

        new_logits.append(h_new)
        new_states.append({"m": m_new, "v": v_new})

    return new_logits, new_states, new_step


def _build_lateral_pairs(lateral_U_list, lateral_log_alpha_list, lateral_force_scale):
    raw_alphas = [jax.nn.softplus(rho) for rho in lateral_log_alpha_list]
    eff_alphas = [lateral_force_scale * a for a in raw_alphas]
    return list(zip(eff_alphas, lateral_U_list))


def _initialize_xs(weights, x_batch, y_onehot):
    xs = [x_batch.T]
    for l in range(1, len(weights)):
        xs.append(weights[l - 1] @ relu(xs[l - 1]))
    xs.append(y_onehot.T)
    return xs


def _output_logit_trajectory(weights, trajectory):
    hidden_traj = relu(trajectory[-2])
    return jnp.einsum("od,dbt->obt", weights[-1], hidden_traj)


def compute_regression_importance(
    weights,
    x_batch,
    y_onehot,
    log_precisions,
    lateral_U_list,
    lateral_log_alpha_list,
    lateral_force_scale,
    num_inference_steps,
    lr_z,
    lambda_dyn=1e-3,
    window=15,
    window_start=None,
    window_end=None,
    num_iters=50,
    standardize_x=False,
    task_il_training=False,
    active_mask=None,
):
    """Compute normalized regression importances for all layers.

    `task_il_training` and `active_mask` select the output-edge error model
    used during inference relaxation. These settings match the main training
    step so pruning gates are fit against the same output model.
    """
    precisions = [jnp.exp(lp) for lp in log_precisions]
    lateral_pairs = _build_lateral_pairs(
        lateral_U_list, lateral_log_alpha_list, lateral_force_scale
    )
    xs_init = _initialize_xs(weights, x_batch, y_onehot)
    _xs_eq, trajectory = infer_with_trajectory(
        weights,
        xs_init,
        precisions,
        lateral_pairs,
        num_inference_steps,
        lr_z,
        task_il_training,
        active_mask,
    )

    raw_importance = []
    for l in range(len(weights)):
        if l == len(weights) - 1:
            child_trajectory = _output_logit_trajectory(weights, trajectory)
        else:
            child_trajectory = trajectory[l + 1]
        d_reg, _window_info = compute_dreg_for_pair(
            trajectory[l],
            child_trajectory,
            window=window,
            window_start=window_start,
            window_end=window_end,
            lambda_dyn=lambda_dyn,
            num_iters=num_iters,
            standardize_x=standardize_x,
        )
        raw_importance.append(d_reg)

    return normalize_importance_list(raw_importance)


def vertical_pruning_stats(
    logits,
    importance,
    layer_scales,
    prune_threshold=0.1,
    match_loss=0.0,
    sparse_loss=0.0,
):
    gates = vertical_gates(logits)
    gates_eff = effective_vertical_gates(logits, layer_scales)
    layers = []
    for l, (h, d, G, G_eff, scale) in enumerate(
        zip(logits, importance, gates, gates_eff, layer_scales)
    ):
        d_flat = jnp.ravel(d)
        layers.append({
            "l": int(l),
            "layer_scale": float(jnp.asarray(scale)),
            "gate_mean": float(jnp.mean(G)),
            "gate_min": float(jnp.min(G)),
            "gate_max": float(jnp.max(G)),
            "gate_std": float(jnp.std(G)),
            "effective_gate_mean": float(jnp.mean(G_eff)),
            "effective_gate_min": float(jnp.min(G_eff)),
            "effective_gate_max": float(jnp.max(G_eff)),
            "logit_mean": float(jnp.mean(h)),
            "logit_std": float(jnp.std(h)),
            "importance_mean": float(jnp.mean(d)),
            "importance_p95": float(_quantile_flat(d_flat, 0.95)),
            "importance_p99": float(_quantile_flat(d_flat, 0.99)),
            "importance_max": float(jnp.max(d)),
            "frac_gate_below_0_5": float(jnp.mean((G < 0.5).astype(jnp.float32))),
            "frac_gate_below_0_1": float(jnp.mean((G < 0.1).astype(jnp.float32))),
            "frac_gate_below_0_01": float(jnp.mean((G < 0.01).astype(jnp.float32))),
            "effective_pruned_frac": float(
                jnp.mean((G_eff < prune_threshold).astype(jnp.float32))
            ),
        })
    return {
        "match_loss": float(jnp.asarray(match_loss)),
        "sparse_loss": float(jnp.asarray(sparse_loss)),
        "prune_threshold": float(prune_threshold),
        "layers": layers,
    }
