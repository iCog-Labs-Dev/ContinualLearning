"""Differentiable 1-step lookahead optimization and anchor evaluation for Meta Causal Coding."""

from typing import Dict, List, Optional, Tuple, Any
import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu
from causal_coding.src.do_influence import compute_meta_causal_gates_from_stats
from causal_coding.src.lateral import lateral_clarity_loss
from causal_coding.src.vertical_pruning import effective_vertical_gates


def decode_omega(
    omega: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Converts unconstrained omega encodings into bounded meta-parameters."""
    beta_lcb = jax.nn.softplus(omega["raw_beta_lcb"])
    tau = 1e-2 * jax.nn.sigmoid(omega["raw_tau"])
    eps_exp = 0.25 * jax.nn.sigmoid(omega["raw_eps_exp"])
    lambda_d = 1e-2 * jax.nn.softplus(omega["raw_lambda_d"])
    protection_strength = 0.1 + 0.75 * jax.nn.sigmoid(omega["raw_protection"])
    return beta_lcb, tau, eps_exp, lambda_d, protection_strength


def _forward_eval(
    weights: List[jnp.ndarray],
    x: jnp.ndarray,
    task_il: bool = False,
    active_classes: Optional[List[int]] = None,
) -> jnp.ndarray:
    """Computes feedforward logits over input batch x."""
    h = x.T
    for l in range(len(weights) - 1):
        h = weights[l] @ relu(h)
    return weights[-1] @ relu(h)


def _compute_classification_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    task_il_training: bool,
    active_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Computes Task-IL active-slice BCE or Class-IL categorical cross-entropy."""
    if task_il_training:
        probs = jax.nn.sigmoid(logits)
        bce = -(targets.T * jnp.log(probs + 1e-8) + (1.0 - targets.T) * jnp.log(1.0 - probs + 1e-8))
        mask = active_mask[:, None] if active_mask.ndim == 1 else active_mask.T
        bce = mask * bce
        return jnp.mean(jnp.sum(bce, axis=0))
    else:
        probs = jax.nn.softmax(logits, axis=0)
        return jnp.mean(-jnp.sum(targets.T * jnp.log(probs + 1e-8), axis=0))


def _compute_candidate_lookahead_weights(
    weights: List[jnp.ndarray],
    gates: List[jnp.ndarray],
    precisions: List[jnp.ndarray],
    errors: List[jnp.ndarray],
    xs: List[jnp.ndarray],
    vertical_eff_gates: List[jnp.ndarray],
    past_support: Optional[List[jnp.ndarray]],
    protection: jnp.ndarray,
    lr_w: float,
    lambda_s: float,
    gate_norm: str = "sum",
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """Computes candidate weights W'(Omega) and synaptic update vectors delta_w."""
    num_layers = len(weights) + 1
    w_primes = []
    delta_ws = []

    for l in range(num_layers - 1):
        weighted_err = precisions[l][:, None] * errors[l + 1]
        per_sample_heb = jnp.einsum("jb,ib->bji", weighted_err, relu(xs[l]))
        gated_heb = gates[l][None, :, :] * per_sample_heb

        if gate_norm == "match":
            raw_norm = jnp.sqrt(jnp.sum(per_sample_heb ** 2, axis=2, keepdims=True) + 1e-12)
            gated_norm = jnp.sqrt(jnp.sum(gated_heb ** 2, axis=2, keepdims=True) + 1e-12)
            scale = jnp.where(gated_norm > 1e-6, raw_norm / gated_norm, 0.0)
            gated_heb = gated_heb * scale

        dw = jnp.mean(gated_heb, axis=0)
        gated_dw = vertical_eff_gates[l] * dw

        if past_support is not None and l < len(past_support) and past_support[l] is not None:
            prot = protection if l < num_layers - 2 else jnp.minimum(0.95, 1.15 * protection)
            gated_dw = gated_dw * (1.0 - prot * past_support[l])

        w_prime = weights[l] + lr_w * gated_dw - lambda_s * jnp.sign(weights[l])
        w_primes.append(w_prime)
        delta_ws.append(gated_dw)

    return w_primes, delta_ws


def _compute_lookahead_confusion_proxy(
    delta_ws: List[jnp.ndarray],
    support_masks: List[jnp.ndarray],
    past_support: Optional[List[jnp.ndarray]],
    num_layers: int,
) -> jnp.ndarray:
    """Computes proxy out-of-support leakage and cross-task overlap cost."""
    leakage_sum = 0.0
    overlap_sum = 0.0

    for l in range(num_layers - 1):
        dw_norm = jnp.sqrt(jnp.sum(delta_ws[l] ** 2) + 1e-12)
        s_curr = support_masks[l]
        leakage_norm = jnp.sqrt(jnp.sum(((1.0 - s_curr) * delta_ws[l]) ** 2) + 1e-12)
        leakage_sum += leakage_norm / dw_norm

        if past_support is not None and past_support[l] is not None:
            overlap_norm = jnp.sqrt(jnp.sum((past_support[l] * delta_ws[l]) ** 2) + 1e-12)
            overlap_sum += overlap_norm / dw_norm

    return 2.0 * (leakage_sum / (num_layers - 1)) + (overlap_sum / (num_layers - 1))


def _compute_clarity_drive_loss(
    params: Dict[str, Any],
    lambda_d: jnp.ndarray,
    confusion_proxy: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluates diffusion clarity potential regularizer on lambda_d."""
    clarity_sum = 0.0
    if "lateral_U" in params and "lateral_log_alpha" in params:
        lat_Us = params["lateral_U"]
        lat_rhos = params["lateral_log_alpha"]
        for U_l, rho_l in zip(lat_Us, lat_rhos):
            alpha_l = jax.nn.softplus(rho_l)
            clarity_sum += lateral_clarity_loss(U_l, alpha_l, t=0.5, eps=1e-4)

    clarity_drive = jax.lax.stop_gradient(confusion_proxy + 0.1 * clarity_sum)
    return clarity_drive * jnp.exp(-100.0 * lambda_d) + 10.0 * (lambda_d ** 2)


def _apply_adam_step(
    params_dict: Dict[str, jnp.ndarray],
    grads_dict: Dict[str, jnp.ndarray],
    adam_state: Dict[str, Any],
    lr: float,
    clip_norm: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """Applies global-norm clipped Adam update to a parameter dictionary."""
    step = adam_state["step"] + 1

    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads_dict.values())) + 1e-8
    clip_scale = jnp.where(grad_norm > clip_norm, clip_norm / grad_norm, 1.0)
    clipped_grads = {k: g * clip_scale for k, g in grads_dict.items()}

    new_params = {}
    new_m = {}
    new_v = {}
    for k in params_dict:
        g = clipped_grads[k]
        m = beta1 * adam_state["m"][k] + (1.0 - beta1) * g
        v = beta2 * adam_state["v"][k] + (1.0 - beta2) * (g ** 2)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        new_params[k] = params_dict[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_m[k] = m
        new_v[k] = v

    new_adam_state = {"m": new_m, "v": new_v, "step": step}
    return new_params, new_adam_state


def sample_anchor_batch(
    anchor_buffer: List[Dict[str, Any]],
    batch_size: int = 64,
    task_il_training: bool = False,
    num_classes: int = 10,
    step_seed: int = 0,
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Samples a balanced anchor batch across previous tasks for meta-evaluation probes."""
    if not anchor_buffer:
        return None, None, None

    total_tasks = len(anchor_buffer)
    per_task = max(1, batch_size // total_tasks)

    Xs, ys, active_masks = [], [], []
    for i, anc in enumerate(anchor_buffer):
        n_avail = anc["X"].shape[0]
        n_take = min(per_task, n_avail)
        perm = jax.random.permutation(jax.random.PRNGKey(step_seed + i), n_avail)[:n_take]
        Xs.append(anc["X"][perm])
        ys.append(anc["y"][perm])

        mask = jnp.zeros(num_classes)
        if "classes" in anc:
            mask = mask.at[jnp.array(anc["classes"])].set(1.0)
        else:
            mask = jnp.ones(num_classes)
        active_masks.append(jnp.tile(mask[None, :], (n_take, 1)))

    cat_X = jnp.concatenate(Xs, axis=0)
    cat_y = jnp.concatenate(ys, axis=0)
    cat_mask = jnp.concatenate(active_masks, axis=0)

    if cat_y.ndim == 1:
        cat_y = jnp.eye(num_classes)[cat_y]

    return cat_X, cat_y, cat_mask


def meta_step(
    model: Any,
    params: Dict[str, Any],
    meta_state: Any,
    curr_X: jnp.ndarray,
    curr_y: jnp.ndarray,
    active_mask: jnp.ndarray,
    task_il_training: bool,
    lr_w: float,
    lambda_s: float,
    gate_p: float,
    gate_kappa: float,
    gate_norm: str = "sum",
    alpha_forget: float = 1.0,
    beta_confusion: float = 0.5,
    lr_meta: float = 1e-2,
    step_seed: int = 0,
) -> Any:
    """Updates meta-parameters omega via 1-step differentiable lookahead on pre-computed equilibrium statistics."""
    weights = params["weights"]
    num_layers = len(weights) + 1
    vertical_gate_logits = params.get("vertical_gate_logits", None)
    vertical_layer_scales = params.get("vertical_layer_scales", None)

    if vertical_gate_logits is not None and vertical_layer_scales is not None:
        vertical_eff_gates = effective_vertical_gates(vertical_gate_logits, vertical_layer_scales)
    else:
        vertical_eff_gates = [jnp.ones_like(w) for w in weights]

    precisions = [jnp.exp(lp) for lp in params["log_precisions"]]
    xs = params["last_xs"]
    errors = params["last_errors"]
    mu_list = params["last_mu_influence"]
    sigma_list = params["last_sigma_influence"]
    past_support = meta_state.consolidated_support_mask if len(meta_state.task_supports) > 0 else None

    anchor_X, anchor_y, anchor_mask = sample_anchor_batch(
        meta_state.anchor_buffer,
        batch_size=min(64, curr_X.shape[0]),
        task_il_training=task_il_training,
        num_classes=weights[-1].shape[0],
        step_seed=step_seed,
    )

    def meta_loss_fn(omega_dict):
        beta_lcb, tau, eps_exp, lambda_d, protection = decode_omega(omega_dict)
        gates, support_masks = compute_meta_causal_gates_from_stats(
            mu_list,
            sigma_list,
            gate_p=gate_p,
            gate_kappa=gate_kappa,
            beta_lcb=beta_lcb,
            tau=tau,
            eps_exp=eps_exp,
            past_support_masks=past_support,
            protection_strength=protection,
        )

        w_primes, delta_ws = _compute_candidate_lookahead_weights(
            weights=weights,
            gates=gates,
            precisions=precisions,
            errors=errors,
            xs=xs,
            vertical_eff_gates=vertical_eff_gates,
            past_support=past_support,
            protection=protection,
            lr_w=lr_w,
            lambda_s=lambda_s,
            gate_norm=gate_norm,
        )

        eff_w_primes = [G * W for W, G in zip(w_primes, vertical_eff_gates)]

        pred_curr = _forward_eval(eff_w_primes, curr_X)
        loss_curr = _compute_classification_loss(
            pred_curr, curr_y, task_il_training=task_il_training, active_mask=active_mask
        )

        if anchor_X is not None and anchor_y is not None:
            pred_anc = _forward_eval(eff_w_primes, anchor_X)
            loss_anchor = _compute_classification_loss(
                pred_anc, anchor_y, task_il_training=task_il_training, active_mask=anchor_mask
            )
        else:
            loss_anchor = 0.0

        confusion_proxy = _compute_lookahead_confusion_proxy(
            delta_ws, support_masks, past_support, num_layers
        )
        lambda_d_reg = _compute_clarity_drive_loss(params, lambda_d, confusion_proxy)

        return (
            loss_curr
            + alpha_forget * loss_anchor
            + beta_confusion * confusion_proxy
            + lambda_d_reg
        )

    grads = jax.grad(meta_loss_fn)(meta_state.omega)
    new_omega, new_adam = _apply_adam_step(
        meta_state.omega, grads, meta_state.adam_state, lr=lr_meta, clip_norm=1.0
    )

    meta_state.omega = new_omega
    meta_state.adam_state = new_adam
    return meta_state

