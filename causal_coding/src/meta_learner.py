"""Online Meta-Learning state tracking, drift detection, and calibration for Causal Coding."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import jax
import jax.numpy as jnp

from causal_coding.src.vertical_pruning import apply_vertical_gates
from causal_coding.src.meta_optimizer import (
    decode_omega,
    sample_anchor_batch,
    meta_step,
    _forward_eval,
    _compute_classification_loss,
)

__all__ = [
    "MetaState",
    "init_meta_state",
    "decode_omega",
    "update_drift_and_detect_shift",
    "calibrate_beta_via_intervention",
    "compute_locality_and_commutator_proxies",
    "compute_pairwise_confusion_matrix",
    "sample_anchor_batch",
    "meta_step",
]


@dataclass
class MetaState:
    """State object for the online Causal Coding Meta-Learner."""
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
    params: Dict[str, Any],
    beta_lcb_init: float = 1.0,
    tau_init: float = 1e-3,
    eps_exp_init: float = 0.05,
    lambda_d_init: float = 1e-3,
    protection_init: float = 0.8,
) -> MetaState:
    """Initializes MetaState with unconstrained parameter encodings and Adam moments."""
    b_val = float(beta_lcb_init)
    raw_beta_lcb = jnp.array(jnp.log(jnp.exp(b_val) - 1.0) if b_val < 20.0 else b_val, dtype=jnp.float32)

    p_tau = jnp.clip(jnp.array(tau_init / 1e-2, dtype=jnp.float32), 1e-4, 1.0 - 1e-4)
    raw_tau = jnp.log(p_tau / (1.0 - p_tau))

    p_eps = jnp.clip(jnp.array(eps_exp_init / 0.25, dtype=jnp.float32), 1e-4, 1.0 - 1e-4)
    raw_eps_exp = jnp.log(p_eps / (1.0 - p_eps))

    ld_val = float(lambda_d_init / 1e-2)
    raw_lambda_d = jnp.array(jnp.log(jnp.exp(ld_val) - 1.0) if ld_val < 20.0 else ld_val, dtype=jnp.float32)

    p_prot = jnp.clip(jnp.array((protection_init - 0.1) / 0.75, dtype=jnp.float32), 1e-4, 1.0 - 1e-4)
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
    return MetaState(
        omega=omega,
        adam_state=adam_state,
        short_influence_ema=[jnp.zeros_like(w) for w in weights],
        long_influence_ema=[jnp.zeros_like(w) for w in weights],
        consolidated_support_mask=[jnp.zeros_like(w) for w in weights],
        task_supports=[],
        task_update_dirs=[],
        anchor_buffer=[],
        confusion_graph=[],
        drift_history=[],
        meta_trajectory=[],
    )


def update_drift_and_detect_shift(
    meta_state: MetaState,
    mu_list: List[jnp.ndarray],
    short_beta: float = 0.8,
    long_beta: float = 0.95,
    shift_threshold: float = 0.35,
) -> Tuple[MetaState, float, bool]:
    """Tracks normalized influence drift D_t across layers and detects mechanism shifts."""
    new_short_ema = []
    new_long_ema = []
    layer_drifts = []

    for l, mu in enumerate(mu_list):
        prev_s = meta_state.short_influence_ema[l]
        prev_l = meta_state.long_influence_ema[l]

        is_init = jnp.all(prev_s == 0.0)
        s_ema = jnp.where(is_init, mu, short_beta * prev_s + (1.0 - short_beta) * mu)
        l_ema = jnp.where(is_init, mu, long_beta * prev_l + (1.0 - long_beta) * mu)

        new_short_ema.append(s_ema)
        new_long_ema.append(l_ema)

        drift_l = jnp.mean(jnp.abs(s_ema - l_ema)) / (jnp.mean(jnp.abs(l_ema)) + 1e-6)
        layer_drifts.append(drift_l)

    drift_stat = float(jnp.max(jnp.stack(layer_drifts)))
    is_shift_detected = bool(drift_stat > shift_threshold)

    meta_state.short_influence_ema = new_short_ema
    meta_state.long_influence_ema = new_long_ema
    meta_state.drift_history.append(drift_stat)

    if is_shift_detected:
        meta_state.long_influence_ema = [
            0.5 * s + 0.5 * l
            for s, l in zip(meta_state.short_influence_ema, meta_state.long_influence_ema)
        ]
        meta_state.omega["raw_eps_exp"] = jnp.maximum(meta_state.omega["raw_eps_exp"], 0.0)

    return meta_state, drift_stat, is_shift_detected


def calibrate_beta_via_intervention(
    model: Any,
    params: Dict[str, Any],
    meta_state: MetaState,
    x_probe: jnp.ndarray,
    y_probe: jnp.ndarray,
    mu_list: List[jnp.ndarray],
    sigma_list: List[jnp.ndarray],
    active_classes: Optional[List[int]] = None,
    target_coverage: float = 0.90,
) -> MetaState:
    """Calibrates beta_lcb online via targeted perturbations of ambiguous synapses."""
    weights = params["weights"]
    vertical_logits = params.get("vertical_gate_logits", None)
    layer_scales = params.get("vertical_layer_scales", None)

    if vertical_logits is not None and layer_scales is not None:
        eff_weights = apply_vertical_gates(weights, vertical_logits, layer_scales)
    else:
        eff_weights = weights

    task_il = active_classes is not None
    num_classes = eff_weights[-1].shape[0]
    active_mask = (
        jnp.zeros(num_classes).at[jnp.array(active_classes)].set(1.0)
        if task_il
        else jnp.ones(num_classes)
    )

    def compute_loss(ws):
        logits = _forward_eval(ws, x_probe)
        return _compute_classification_loss(
            logits, y_probe, task_il_training=task_il, active_mask=active_mask
        )

    base_loss = compute_loss(eff_weights)
    _, tau, _, _, _ = decode_omega(meta_state.omega)

    ratios = []
    for l in range(len(eff_weights)):
        mu = mu_list[l]
        sigma = sigma_list[l]

        ambiguity = sigma / (jnp.abs(mu - tau) + 1e-4)
        flat_amb = ambiguity.flatten()
        k = min(32, flat_amb.shape[0])
        thresh = jnp.sort(flat_amb)[-k]
        mask_intervene = (ambiguity >= thresh)

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
    """Computes locality leakage eps_c, commutator defect delta_comm, and total confusion cost."""
    eps_c_list = []
    delta_comm_list = []

    for l, dw in enumerate(delta_weights):
        dw_norm = jnp.linalg.norm(dw) + 1e-8
        s_curr = current_support_masks[l]
        s_past = consolidated_support_mask[l]

        leakage = jnp.linalg.norm((1.0 - s_curr) * dw) / dw_norm
        eps_c_list.append(leakage)

        overlap = jnp.linalg.norm(s_past * dw) / dw_norm
        delta_comm_list.append(overlap)

    eps_c = float(jnp.mean(jnp.array(eps_c_list)))
    delta_comm = float(jnp.mean(jnp.array(delta_comm_list)))

    if len(past_task_update_dirs) > 0:
        cossim_sqs = []
        total_curr_norm = jnp.sqrt(sum(jnp.sum(dw ** 2) for dw in delta_weights)) + 1e-8
        curr_dir = [dw / total_curr_norm for dw in delta_weights]

        for prev_dir in past_task_update_dirs:
            cossim = sum(jnp.sum(cd * pd) for cd, pd in zip(curr_dir, prev_dir))
            cossim_sqs.append(cossim ** 2)
        delta_comm += float(jnp.mean(jnp.array(cossim_sqs)))

    confusion_cost = 2.0 * eps_c + delta_comm
    return eps_c, delta_comm, confusion_cost


def compute_pairwise_confusion_matrix(
    task_supports: List[List[jnp.ndarray]],
    task_update_dirs: List[List[jnp.ndarray]],
    consolidated_mask: Optional[List[jnp.ndarray]] = None,
) -> List[List[float]]:
    """Computes the full T x T pairwise confusion matrix across completed tasks."""
    T = len(task_supports)
    if T == 0:
        return []

    conf_matrix = jnp.zeros((T, T))
    for i in range(T):
        for j in range(T):
            if i == j:
                continue

            eps_i = jnp.mean(jnp.array([
                jnp.linalg.norm((1.0 - task_supports[i][l]) * task_update_dirs[i][l])
                for l in range(len(task_supports[i]))
            ]))
            eps_j = jnp.mean(jnp.array([
                jnp.linalg.norm((1.0 - task_supports[j][l]) * task_update_dirs[j][l])
                for l in range(len(task_supports[j]))
            ]))

            overlap_ij = jnp.mean(jnp.array([
                jnp.linalg.norm(task_supports[j][l] * task_update_dirs[i][l])
                for l in range(len(task_supports[i]))
            ]))
            cos_ij = sum(
                jnp.sum(task_update_dirs[i][l] * task_update_dirs[j][l])
                for l in range(len(task_supports[i]))
            )
            delta_ij = overlap_ij + (cos_ij ** 2)

            conf_matrix = conf_matrix.at[i, j].set(2.0 * (eps_i + eps_j) + delta_ij)

    return conf_matrix.tolist()
