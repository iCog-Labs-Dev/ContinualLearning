from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu
from causal_coding.src.do_influence import (
    compute_meta_causal_gates_per_sample,
    compute_meta_causal_gates_from_stats,
)
from causal_coding.src.lateral import lateral_clarity_loss
from causal_coding.src.vertical_pruning import apply_vertical_gates, effective_vertical_gates


@dataclass
class MetaState:
    """State object for the online Causal Coding Meta-Learner (Goertzel 2025)."""
    omega: Dict[str, jnp.ndarray]
    adam_state: Dict[str, Any]
    short_influence_ema: List[jnp.ndarray]
    long_influence_ema: List[jnp.ndarray]
    consolidated_support_mask: List[jnp.ndarray]
    task_supports: List[List[jnp.ndarray]] = field(default_factory=list)
    task_update_dirs: List[List[jnp.ndarray]] = field(default_factory=list)
    anchor_buffer: List[Dict[str, Any]] = field(default_factory=list)
    confusion_graph: List[List[float]] = field(default_factory=list)
    drift_history: List[float] = field(default_factory=list)
    meta_trajectory: List[Dict[str, float]] = field(default_factory=list)


def init_meta_state(
    params,
    beta_lcb_init=1.0,
    tau_init=1e-3,
    eps_exp_init=0.05,
    lambda_d_init=1e-3,
    protection_init=0.8,
) -> MetaState:
    """Initialize MetaState matching params['weights'] shapes."""
    # Convert init values into unconstrained parameter representations:
    # 1. beta_lcb = softplus(raw_beta_lcb) -> raw = log(exp(beta) - 1)
    b_val = float(beta_lcb_init)
    raw_beta_lcb = jnp.array(jnp.log(jnp.exp(b_val) - 1.0) if b_val < 20.0 else b_val, dtype=jnp.float32)

    # 2. tau = 1e-2 * sigmoid(raw_tau) -> raw = logit(tau / 1e-2)
    p_tau = jnp.clip(jnp.array(tau_init / 1e-2, dtype=jnp.float32), 1e-4, 1.0 - 1e-4)
    raw_tau = jnp.log(p_tau / (1.0 - p_tau))

    # 3. eps_exp = 0.25 * sigmoid(raw_eps_exp) -> raw = logit(eps_exp / 0.25)
    p_eps = jnp.clip(jnp.array(eps_exp_init / 0.25, dtype=jnp.float32), 1e-4, 1.0 - 1e-4)
    raw_eps_exp = jnp.log(p_eps / (1.0 - p_eps))

    # 4. lambda_d = 1e-2 * softplus(raw_lambda_d) -> raw = log(exp(lambda_d / 1e-2) - 1)
    ld_val = float(lambda_d_init / 1e-2)
    raw_lambda_d = jnp.array(jnp.log(jnp.exp(ld_val) - 1.0) if ld_val < 20.0 else ld_val, dtype=jnp.float32)

    # 5. protection = sigmoid(raw_protection) -> raw = logit(protection)
    p_prot = jnp.clip(jnp.array(protection_init, dtype=jnp.float32), 1e-4, 1.0 - 1e-4)
    raw_protection = jnp.log(p_prot / (1.0 - p_prot))

    omega = {
        "raw_beta_lcb": raw_beta_lcb,
        "raw_tau": raw_tau,
        "raw_eps_exp": raw_eps_exp,
        "raw_lambda_d": raw_lambda_d,
        "raw_protection": raw_protection,
    }

    adam_state = {
        "m": {k: jnp.zeros_like(v) for k, v in omega.items()},
        "v": {k: jnp.zeros_like(v) for k, v in omega.items()},
        "step": 0,
    }

    weights = params["weights"]
    short_ema = [jnp.zeros_like(w) for w in weights]
    long_ema = [jnp.zeros_like(w) for w in weights]
    consolidated = [jnp.zeros_like(w) for w in weights]

    return MetaState(
        omega=omega,
        adam_state=adam_state,
        short_influence_ema=short_ema,
        long_influence_ema=long_ema,
        consolidated_support_mask=consolidated,
        task_supports=[],
        task_update_dirs=[],
        anchor_buffer=[],
        confusion_graph=[],
        drift_history=[],
        meta_trajectory=[],
    )


def decode_omega(omega: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Converts unconstrained omega into bounded meta-parameters."""
    beta_lcb = jax.nn.softplus(omega["raw_beta_lcb"])
    tau = 1e-2 * jax.nn.sigmoid(omega["raw_tau"])
    eps_exp = 0.25 * jax.nn.sigmoid(omega["raw_eps_exp"])
    lambda_d = 1e-2 * jax.nn.softplus(omega["raw_lambda_d"])
    protection_strength = jax.nn.sigmoid(omega["raw_protection"])
    return beta_lcb, tau, eps_exp, lambda_d, protection_strength


def update_drift_and_detect_shift(
    meta_state: MetaState,
    mu_list: List[jnp.ndarray],
    short_beta: float = 0.8,
    long_beta: float = 0.95,
    shift_threshold: float = 0.15,
) -> Tuple[MetaState, float, bool]:
    """Tracks influence drift D_t and detects mechanism change-points (Section 2.4)."""
    new_short_ema = []
    new_long_ema = []
    layer_drifts = []

    for l, mu in enumerate(mu_list):
        prev_s = meta_state.short_influence_ema[l]
        prev_l = meta_state.long_influence_ema[l]

        # Initialize EMAs on first non-zero update
        is_init = jnp.all(prev_s == 0.0)
        s_ema = jnp.where(is_init, mu, short_beta * prev_s + (1.0 - short_beta) * mu)
        l_ema = jnp.where(is_init, mu, long_beta * prev_l + (1.0 - long_beta) * mu)

        new_short_ema.append(s_ema)
        new_long_ema.append(l_ema)

        drift_l = jnp.max(jnp.abs(s_ema - l_ema))
        layer_drifts.append(drift_l)

    drift_stat = float(jnp.max(jnp.stack(layer_drifts)))
    is_shift_detected = bool(drift_stat > shift_threshold)

    meta_state.short_influence_ema = new_short_ema
    meta_state.long_influence_ema = new_long_ema
    meta_state.drift_history.append(drift_stat)

    if is_shift_detected:
        # Mechanism shift detected: reset short EMA to adapt immediately
        meta_state.short_influence_ema = [jnp.array(mu) for mu in mu_list]
        # Boost exploration parameter slightly to probe new mechanism
        meta_state.omega["raw_eps_exp"] = jnp.maximum(meta_state.omega["raw_eps_exp"], 0.0)

    return meta_state, drift_stat, is_shift_detected


def _forward_eval(weights, x, task_il=False, active_classes=None):
    """Simple feedforward prediction loss."""
    h = x.T  # (d_in, B)
    for l in range(len(weights) - 1):
        h = weights[l] @ relu(h)
    logits = weights[-1] @ relu(h)  # (C, B)
    return logits


def calibrate_beta_via_intervention(
    model,
    params,
    meta_state: MetaState,
    x_probe: jnp.ndarray,
    y_probe: jnp.ndarray,
    mu_list: List[jnp.ndarray],
    sigma_list: List[jnp.ndarray],
    active_classes: Optional[List[int]] = None,
    target_coverage: float = 0.90,
) -> MetaState:
    """Calibrate beta_lcb online via targeted interventions on ambiguous synapses (Section 3.3)."""
    weights = params["weights"]
    vertical_logits = params.get("vertical_gate_logits", None)
    layer_scales = params.get("vertical_layer_scales", None)

    if vertical_logits is not None and layer_scales is not None:
        eff_weights = apply_vertical_gates(weights, vertical_logits, layer_scales)
    else:
        eff_weights = weights

    task_il = active_classes is not None
    num_classes = eff_weights[-1].shape[0]

    if task_il:
        active_mask = jnp.zeros(num_classes).at[jnp.array(active_classes)].set(1.0)
    else:
        active_mask = jnp.ones(num_classes)

    def compute_loss(ws):
        logits = _forward_eval(ws, x_probe, task_il=task_il)
        if task_il:
            p = jax.nn.sigmoid(logits)
            bce = -(y_probe.T * jnp.log(p + 1e-8) + (1.0 - y_probe.T) * jnp.log(1.0 - p + 1e-8))
            bce = active_mask[:, None] * bce
            return jnp.mean(jnp.sum(bce, axis=0))
        else:
            probs = jax.nn.softmax(logits, axis=0)
            ce = -jnp.sum(y_probe.T * jnp.log(probs + 1e-8), axis=0)
            return jnp.mean(ce)

    base_loss = compute_loss(eff_weights)
    _, tau, _, _, _ = decode_omega(meta_state.omega)

    ratios = []
    for l in range(len(eff_weights)):
        mu = mu_list[l]
        sigma = sigma_list[l]

        # Ambiguous / uncertain synapses: high uncertainty or near threshold tau
        ambiguity = sigma / (jnp.abs(mu - tau) + 1e-4)
        flat_amb = ambiguity.flatten()
        k = min(32, flat_amb.shape[0])
        thresh = jnp.sort(flat_amb)[-k]
        mask_intervene = (ambiguity >= thresh)

        # Intervene by zeroing out ambiguous synapses in layer l
        ws_pert = list(eff_weights)
        ws_pert[l] = eff_weights[l] * (1.0 - mask_intervene.astype(jnp.float32))

        pert_loss = compute_loss(ws_pert)
        rel_emp_gap = jnp.abs(pert_loss - base_loss) / (jnp.abs(base_loss) + 1e-4)

        rel_pred_mu = jnp.sum(mu * mask_intervene) / (jnp.sum(mu) + 1e-8)
        rel_sigma = jnp.sum(sigma * mask_intervene) / (jnp.sum(mu) + 1e-8)

        ratio = jnp.abs(rel_emp_gap - rel_pred_mu) / (rel_sigma + 1e-4)
        ratios.append(ratio)

    mean_ratio = jnp.mean(jnp.array(ratios))
    target_beta = jnp.clip(1.0 + 0.5 * (mean_ratio - 1.0), 0.5, 3.0)

    # Smoothly nudge raw_beta_lcb toward target_beta
    curr_beta = jax.nn.softplus(meta_state.omega["raw_beta_lcb"])
    updated_beta = 0.85 * curr_beta + 0.15 * target_beta
    meta_state.omega["raw_beta_lcb"] = jnp.log(jnp.exp(updated_beta) - 1.0)

    return meta_state


def compute_locality_and_commutator_proxies(
    delta_weights: List[jnp.ndarray],
    current_support_masks: List[jnp.ndarray],
    consolidated_support_mask: List[jnp.ndarray],
    past_task_update_dirs: List[List[jnp.ndarray]],
) -> Tuple[float, float, float]:
    """Computes locality leakage eps_c, commutator defect delta_comm, and confusion cost (Section 3.2 & 3.5)."""
    eps_c_list = []
    delta_comm_list = []

    for l, dw in enumerate(delta_weights):
        dw_norm = jnp.linalg.norm(dw) + 1e-8
        s_curr = current_support_masks[l]
        s_past = consolidated_support_mask[l]

        # Locality error: updates outside current support
        leakage = jnp.linalg.norm((1.0 - s_curr) * dw) / dw_norm
        eps_c_list.append(leakage)

        # Commutator defect: overlap with past consolidated supports
        overlap = jnp.linalg.norm(s_past * dw) / dw_norm
        delta_comm_list.append(overlap)

    eps_c = float(jnp.mean(jnp.array(eps_c_list)))
    delta_comm = float(jnp.mean(jnp.array(delta_comm_list)))

    # Additional cosine similarity defect if past task directions exist
    if len(past_task_update_dirs) > 0:
        cossim_sqs = []
        # Normalize current delta_weights
        total_curr_norm = jnp.sqrt(sum(jnp.sum(dw ** 2) for dw in delta_weights)) + 1e-8
        curr_dir = [dw / total_curr_norm for dw in delta_weights]

        for prev_dir in past_task_update_dirs:
            cossim = sum(jnp.sum(cd * pd) for cd, pd in zip(curr_dir, prev_dir))
            cossim_sqs.append(cossim ** 2)
        delta_comm += float(jnp.mean(jnp.array(cossim_sqs)))

    confusion_cost = 2.0 * eps_c + delta_comm
    return eps_c, delta_comm, confusion_cost


def sample_anchor_batch(
    anchor_buffer: List[Dict[str, Any]],
    batch_size: int = 64,
    task_il_training: bool = False,
    num_classes: int = 10,
    step_seed: int = 0,
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Samples a balanced anchor batch across previous tasks."""
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

    # If anchor y is 1D integer, convert to one-hot
    if cat_y.ndim == 1:
        cat_y = jnp.eye(num_classes)[cat_y]

    return cat_X, cat_y, cat_mask


def meta_step(
    model,
    params,
    meta_state: MetaState,
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
) -> MetaState:
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

    # Sample anchors if available
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

        w_primes = []
        delta_ws = []
        for l in range(num_layers - 1):
            weighted_err = precisions[l][:, None] * errors[l + 1]
            per_sample_heb = jnp.einsum("jb,ib->bji", weighted_err, relu(xs[l]))
            gated_heb = gates[l][None, :, :] * per_sample_heb
            if gate_norm == "match":
                raw_norm = jnp.sqrt(jnp.sum(per_sample_heb ** 2, axis=2, keepdims=True) + 1e-12)
                gated_norm = jnp.sqrt(jnp.sum(gated_heb ** 2, axis=2, keepdims=True) + 1e-12)
                scale = jnp.where(
                    gated_norm > 1e-6, raw_norm / gated_norm, 0.0
                )
                gated_heb = gated_heb * scale
            dw = jnp.mean(gated_heb, axis=0)
            gated_dw = vertical_eff_gates[l] * dw
            if past_support is not None and l < len(past_support) and past_support[l] is not None:
                gated_dw = gated_dw * (1.0 - protection * past_support[l])
            w_prime = weights[l] + lr_w * gated_dw - lambda_s * jnp.sign(weights[l])
            w_primes.append(w_prime)
            delta_ws.append(gated_dw)

        # 1. Current batch loss under w_primes
        pred_curr = _forward_eval(w_primes, curr_X)

        if task_il_training:
            p_curr = jax.nn.sigmoid(pred_curr)
            bce_curr = -(curr_y.T * jnp.log(p_curr + 1e-8) + (1.0 - curr_y.T) * jnp.log(1.0 - p_curr + 1e-8))
            bce_curr = active_mask[:, None] * bce_curr
            loss_curr = jnp.mean(jnp.sum(bce_curr, axis=0))
        else:
            probs_curr = jax.nn.softmax(pred_curr, axis=0)
            loss_curr = jnp.mean(-jnp.sum(curr_y.T * jnp.log(probs_curr + 1e-8), axis=0))

        # 2. Anchor forgetting loss
        if anchor_X is not None and anchor_y is not None:
            pred_anc = _forward_eval(w_primes, anchor_X)

            if task_il_training:
                p_anc = jax.nn.sigmoid(pred_anc)
                bce_anc = -(anchor_y.T * jnp.log(p_anc + 1e-8) + (1.0 - anchor_y.T) * jnp.log(1.0 - p_anc + 1e-8))
                bce_anc = anchor_mask.T * bce_anc
                loss_anchor = jnp.mean(jnp.sum(bce_anc, axis=0))
            else:
                probs_anc = jax.nn.softmax(pred_anc, axis=0)
                loss_anchor = jnp.mean(-jnp.sum(anchor_y.T * jnp.log(probs_anc + 1e-8), axis=0))
        else:
            loss_anchor = 0.0

        # 3. Locality and Commutator defect proxy
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

        confusion_proxy = 2.0 * (leakage_sum / (num_layers - 1)) + (overlap_sum / (num_layers - 1))

        # 4. Connect lambda_d to lateral clarity and confusion proxy (Fix 3)
        clarity_sum = 0.0
        if "lateral_U" in params and "lateral_log_alpha" in params:
            lat_Us = params["lateral_U"]
            lat_rhos = params["lateral_log_alpha"]
            for U_l, rho_l in zip(lat_Us, lat_rhos):
                alpha_l = jax.nn.softplus(rho_l)
                clarity_sum += lateral_clarity_loss(U_l, alpha_l, t=0.5, eps=1e-4)

        clarity_drive = jax.lax.stop_gradient(confusion_proxy + 0.1 * clarity_sum)
        lambda_d_reg = (
            clarity_drive * jnp.exp(-100.0 * lambda_d)
            + 10.0 * (lambda_d ** 2)
        )

        # Total Meta-Objective (Section 3.5 & Fix 3)
        total_meta_loss = (
            loss_curr
            + alpha_forget * loss_anchor
            + beta_confusion * confusion_proxy
            + lambda_d_reg
        )
        return total_meta_loss

    grads = jax.grad(meta_loss_fn)(meta_state.omega)

    # Hand-rolled Adam update on omega
    adam = meta_state.adam_state
    step = adam["step"] + 1
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Global gradient clipping
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads.values())) + 1e-8
    clip_scale = jnp.where(grad_norm > 1.0, 1.0 / grad_norm, 1.0)
    grads = {k: g * clip_scale for k, g in grads.items()}

    new_omega = {}
    new_m = {}
    new_v = {}
    for k in meta_state.omega:
        g = grads[k]
        m = beta1 * adam["m"][k] + (1.0 - beta1) * g
        v = beta2 * adam["v"][k] + (1.0 - beta2) * (g ** 2)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        new_omega[k] = meta_state.omega[k] - lr_meta * m_hat / (jnp.sqrt(v_hat) + eps)
        new_m[k] = m
        new_v[k] = v

    meta_state.omega = new_omega
    meta_state.adam_state = {"m": new_m, "v": new_v, "step": step}
    return meta_state
