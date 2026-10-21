import jax
import jax.numpy as jnp
from functools import partial

from causal_coding.src.activations import relu
from causal_coding.src.inference import _infer_step, compute_errors
from causal_coding.src.do_influence import (
    compute_jacobians,
    compute_jacobians_per_sample,
    compute_local_maps,
    compute_causal_gates_per_sample,
    compute_output_fisher,
)
from causal_coding.src.lateral import (
    total_lateral_loss,
    adam_update_lateral,
    apply_spectral_cap,
    update_cov_ema,
)
from causal_coding.src.activations import relu_derivative
from causal_coding.src.lateral_ops import build_lateral_ops, struct_update
from causal_coding.src.vertical_pruning import (
    adam_update_vertical,
    apply_vertical_gates,
    effective_vertical_gates,
    vertical_gate_loss_components,
)


@partial(jax.jit, static_argnums=(3, 4, 45, 46, 47, 48, 49))
def train_step(
    params,
    x_batch,
    y_batch,
    num_inference_steps,
    k_probe,
    lr_z,
    lr_w,
    gate_p,
    gate_kappa,
    ridge,
    lambda_s,
    beta_pi,
    lateral_force_scale,
    lateral_lr_scale,
    lr_lat,
    beta_cov,
    eps_lat,
    beta_logdet,
    lambda_fro,
    lambda_U,
    lambda_max_cap,
    adam_beta1_lat,
    adam_beta2_lat,
    adam_eps_lat,
    grad_clip_norm_lat,
    lambda_d,
    clarity_t,
    clarity_eps,
    vertical_lr_scale,
    vertical_layer_scales,
    lr_vert,
    vertical_alpha_g,
    lambda_vert_match,
    lambda_vert_sparse,
    vertical_eps,
    adam_beta1_vert,
    adam_beta2_vert,
    adam_eps_vert,
    grad_clip_norm_vert,
    pi0,
    rho_v,
    delta_abs,
    d_min,
    d_max,
    active_mask=None,
    task_il_training=False,
    gate_norm="sum",
    lateral_mode="lowrank",            # static 47
    struct_learning="fixed",           # static 48
    schur_include_structured=False,    # static 49
    struct_lr_scale=1.0,               # traced 50
    struct_lr_s=1e-3,                  # traced 51
    struct_lr_b=1e-3,                  # traced 52
    struct_rho=0.2,                    # traced 53
    struct_beta_stat=0.99,             # traced 54
    struct_guard_c=0.5,                # traced 55
):
    weights = params["weights"]
    log_precisions = params["log_precisions"]
    precision_var_ema = params["precision_var_ema"]
    lateral_U_list = params["lateral_U"]
    lateral_log_alpha_list = params["lateral_log_alpha"]
    lateral_cov_ema_list = params["lateral_cov_ema"]
    lateral_adam_states = params["lateral_adam_states"]
    lateral_adam_step = params["lateral_adam_step"]
    vertical_gate_logits = params["vertical_gate_logits"]
    vertical_importance = params["vertical_importance"]
    vertical_adam_states = params["vertical_adam_states"]
    vertical_adam_step = params["vertical_adam_step"]

    num_layers = len(weights) + 1
    num_hidden = num_layers - 2
    precisions = [jnp.exp(lp) for lp in log_precisions]

    lateral_pairs = build_lateral_ops(params, lateral_mode, lateral_force_scale)

    # Network computations use vertically gated effective weights.
    effective_weights = apply_vertical_gates(
        weights, vertical_gate_logits, vertical_layer_scales
    )
    vertical_eff_gates = effective_vertical_gates(
        vertical_gate_logits, vertical_layer_scales
    )

    # Feedforward init + label clamp.
    xs = [x_batch.T]
    for l in range(1, num_layers - 1):
        xs.append(effective_weights[l - 1] @ relu(xs[l - 1]))
    xs.append(y_batch.T)

    # Early fixed inference steps with residual capture for precision EMA.
    early_eps_sq_sum = [jnp.zeros_like(xs[l + 1]) for l in range(num_hidden)]
    for _ in range(k_probe):
        xs = _infer_step(
            effective_weights, xs, precisions, lateral_pairs, lr_z,
            task_il_training, active_mask,
        )
        for l in range(num_hidden):
            pred = effective_weights[l] @ relu(xs[l])
            eps = xs[l + 1] - pred
            early_eps_sq_sum[l] = early_eps_sq_sum[l] + eps ** 2

    remaining_steps = jnp.maximum(num_inference_steps - k_probe, 0)

    def body_fn(_, xs_inner):
        return _infer_step(
            effective_weights, xs_inner, precisions, lateral_pairs, lr_z,
            task_il_training, active_mask,
        )

    xs = jax.lax.fori_loop(0, remaining_steps, body_fn, xs)
    errors = compute_errors(effective_weights, xs, task_il_training, active_mask)

    jacobians = compute_jacobians(effective_weights, xs)
    jacobians_per_sample = compute_jacobians_per_sample(effective_weights, xs)
    output_fisher = compute_output_fisher(
        effective_weights, xs, task_il_training, active_mask
    )
    hidden_masks = [
        jnp.mean(relu_derivative(xs[l + 1]), axis=1) for l in range(num_hidden)
    ]
    A_tildes = compute_local_maps(
        precisions,
        lateral_pairs,
        jacobians,
        ridge,
        output_fisher=output_fisher,
        jacobians_per_sample=jacobians_per_sample,
        hidden_masks=hidden_masks,
        include_structured=schur_include_structured,
    )
    # Apply per-sample gates to per-sample Hebbian outer products, then average
    # across the batch.
    gates_per_sample = compute_causal_gates_per_sample(A_tildes, gate_p, gate_kappa)

    def vertical_loss_fn(logits):
        total, _match, _sparse = vertical_gate_loss_components(
            logits,
            vertical_importance,
            vertical_alpha_g,
            lambda_vert_match,
            lambda_vert_sparse,
            vertical_eps,
            vertical_layer_scales,
        )
        return total

    _vertical_total_loss, vertical_match_loss, vertical_sparse_loss = (
        vertical_gate_loss_components(
            vertical_gate_logits,
            vertical_importance,
            vertical_alpha_g,
            lambda_vert_match,
            lambda_vert_sparse,
            vertical_eps,
            vertical_layer_scales,
        )
    )
    grad_vertical_logits = jax.grad(vertical_loss_fn)(vertical_gate_logits)
    (
        new_vertical_gate_logits,
        new_vertical_adam_states,
        new_vertical_adam_step,
    ) = adam_update_vertical(
        vertical_gate_logits,
        grad_vertical_logits,
        vertical_adam_states,
        vertical_adam_step,
        lr_vert,
        vertical_lr_scale,
        vertical_layer_scales,
        beta1=adam_beta1_vert,
        beta2=adam_beta2_vert,
        eps=adam_eps_vert,
        clip_norm=grad_clip_norm_vert,
    )

    new_weights = []
    for l in range(num_layers - 1):
        weighted_error = precisions[l][:, None] * errors[l + 1]
        # Per-sample Hebbian update, averaged across the batch.
        # Uses the per-sample local update; the outer product
        # materialises (B, d_{l+1}, d_l), so memory scales with batch and
        # layer width.
        per_sample_hebbian = jnp.einsum(
            "jb,ib->bji", weighted_error, relu(xs[l])
        )
        gated_hebbian = gates_per_sample[l] * per_sample_hebbian
        if gate_norm == "match":
            # Preserve each per-sample child update norm after gating.
            # Zero-row guard avoids division by near-zero norms.
            raw_norm = jnp.linalg.norm(per_sample_hebbian, axis=2, keepdims=True)
            gated_norm = jnp.linalg.norm(gated_hebbian, axis=2, keepdims=True)
            scale = jnp.where(
                gated_norm > 1e-12, raw_norm / (gated_norm + 1e-12), 0.0
            )
            gated_hebbian = gated_hebbian * scale
        delta_w = jnp.mean(gated_hebbian, axis=0)
        gated_delta_w = vertical_eff_gates[l] * delta_w

        cand_w = weights[l] + lr_w * gated_delta_w - lambda_s * jnp.sign(weights[l])
        if l < num_layers - 2:
            # Per-neuron (row-wise) L2 norm cap: initial He row norm is sqrt(2.0) ≈ 1.414.
            # Capping per-row ensures active neurons on Task 2-5 never shrink Task 1 neurons!
            row_norms = jnp.linalg.norm(cand_w, axis=1, keepdims=True)
            max_row_norm = jnp.sqrt(2.0) * 1.05
            cand_w = jnp.where(
                row_norms > max_row_norm,
                cand_w * (max_row_norm / (row_norms + 1e-12)),
                cand_w,
            )
        new_w = jnp.where(jnp.isfinite(cand_w), cand_w, weights[l])
        new_weights.append(new_w)

    # Update hidden-layer residual precision from early-inference residual
    # variance. Structured relative-D diagonal precision; output precision stays
    # frozen at zero.
    new_precision_var_ema = []
    new_log_precisions = []
    log_pi0 = jnp.log(pi0)
    for l in range(num_hidden):
        mean_batch_eps_sq = jnp.mean(early_eps_sq_sum[l] / k_probe, axis=1)
        new_v = beta_pi * precision_var_ema[l] + (1.0 - beta_pi) * mean_batch_eps_sq
        v_bar = jnp.mean(new_v)
        delta_l = rho_v * v_bar + delta_abs
        d_tilde = (v_bar + delta_l) / (new_v + delta_l)
        d_clip = jnp.clip(d_tilde, d_min, d_max)
        d_l = d_clip / jnp.mean(d_clip)
        new_log_pi = log_pi0 + jnp.log(d_l)
        new_precision_var_ema.append(new_v)
        new_log_precisions.append(new_log_pi)
    new_log_precisions.append(log_precisions[-1])

    if lateral_mode in ("lowrank", "both"):
        # Update hidden-state covariance estimates and lateral precision parameters.
        new_cov_emas = []
        for l_idx in range(num_hidden):
            z_l_eq = xs[l_idx + 1]
            new_cov_emas.append(
                update_cov_ema(lateral_cov_ema_list[l_idx], z_l_eq, beta_cov)
            )

        cov_emas_stopgrad = [jax.lax.stop_gradient(C) for C in new_cov_emas]

        def lateral_loss_fn(Us, rhos):
            return total_lateral_loss(
                Us,
                rhos,
                cov_emas_stopgrad,
                lateral_force_scale,
                beta_logdet,
                eps_lat,
                lambda_fro,
                lambda_U,
                lambda_d,
                clarity_t,
                clarity_eps,
            )

        grad_Us, grad_rhos = jax.grad(lateral_loss_fn, argnums=(0, 1))(
            lateral_U_list, lateral_log_alpha_list
        )

        new_lateral_U, new_lateral_rho, new_lateral_adam_states, new_lateral_adam_step = (
            adam_update_lateral(
                lateral_U_list,
                lateral_log_alpha_list,
                grad_Us,
                grad_rhos,
                lateral_adam_states,
                lateral_adam_step,
                lr_lat,
                lateral_lr_scale,
                beta1=adam_beta1_lat,
                beta2=adam_beta2_lat,
                eps=adam_eps_lat,
                clip_norm=grad_clip_norm_lat,
            )
        )

        new_lateral_U = [
            apply_spectral_cap(U, rho, lambda_max_cap)
            for U, rho in zip(new_lateral_U, new_lateral_rho)
        ]

    else:
        # Structured-only: low-rank state passes through unchanged.
        new_lateral_U = params["lateral_U"]
        new_lateral_rho = params["lateral_log_alpha"]
        new_cov_emas = params["lateral_cov_ema"]
        new_lateral_adam_states = params["lateral_adam_states"]
        new_lateral_adam_step = params["lateral_adam_step"]

    # Structured lateral: local rule, computed after the precision update so
    # the guard sees the precisions the next step will run with.
    if lateral_mode in ("structured", "both"):
        new_struct = [
            struct_update(
                params["struct_s"][l], params["struct_b"][l],
                params["struct_p_ema"][l], params["struct_q_ema"][l],
                xs[l + 1], jnp.exp(new_log_precisions[l]),
                struct_lr_s, struct_lr_b, struct_rho, struct_beta_stat,
                struct_guard_c, struct_learning, struct_lr_scale,
            )
            for l in range(num_hidden)
        ]

    new_params = {
        "weights": new_weights,
        "log_precisions": new_log_precisions,
        "precision_var_ema": new_precision_var_ema,
        "lateral_U": new_lateral_U,
        "lateral_log_alpha": new_lateral_rho,
        "lateral_cov_ema": new_cov_emas,
        "lateral_adam_states": new_lateral_adam_states,
        "lateral_adam_step": new_lateral_adam_step,
        "vertical_gate_logits": new_vertical_gate_logits,
        "vertical_importance": vertical_importance,
        "vertical_adam_states": new_vertical_adam_states,
        "vertical_adam_step": new_vertical_adam_step,
        "vertical_layer_scales": [
            jnp.asarray(scale, dtype=weights[0].dtype) for scale in vertical_layer_scales
        ],
        "vertical_match_loss": vertical_match_loss,
        "vertical_sparse_loss": vertical_sparse_loss,
    }
    if lateral_mode in ("structured", "both"):
        new_params.update({
            "struct_s": [r[0] for r in new_struct],
            "struct_b": [r[1] for r in new_struct],
            "struct_p_ema": [r[2] for r in new_struct],
            "struct_q_ema": [r[3] for r in new_struct],
            "struct_guard_frac": [r[4] for r in new_struct],
        })

    new_effective_weights = apply_vertical_gates(
        new_weights, new_vertical_gate_logits, vertical_layer_scales
    )
    prediction = new_effective_weights[-1] @ relu(xs[-2])
    if task_il_training:
        # Active-slice BCE: sum over active classes, then average over samples.
        p = jax.nn.sigmoid(prediction)
        bce = -(
            y_batch.T * jnp.log(jnp.clip(p, 1e-7, 1.0 - 1e-7))
            + (1.0 - y_batch.T) * jnp.log(jnp.clip(1.0 - p, 1e-7, 1.0 - 1e-7))
        )
        bce = active_mask[:, None] * bce
        loss = jnp.mean(jnp.sum(bce, axis=0))
        loss = jnp.where(jnp.isfinite(loss), loss, 0.0)
    else:
        probs = jax.nn.softmax(prediction, axis=0)
        log_probs = jnp.log(jnp.clip(probs, 1e-7, 1.0))
        ce = -jnp.sum(y_batch.T * log_probs, axis=0)
        loss = jnp.mean(ce)
        loss = jnp.where(jnp.isfinite(loss), loss, 0.0)

    return new_params, loss
