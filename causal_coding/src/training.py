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
from causal_coding.src.vertical_pruning import (
    adam_update_vertical,
    apply_vertical_gates,
    effective_vertical_gates,
    vertical_gate_loss_components,
)


@partial(jax.jit, static_argnums=(3, 4))
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

    raw_alphas = [jax.nn.softplus(rho) for rho in lateral_log_alpha_list]
    eff_alphas = [lateral_force_scale * a for a in raw_alphas]
    lateral_pairs = list(zip(eff_alphas, lateral_U_list))

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
        xs = _infer_step(effective_weights, xs, precisions, lateral_pairs, lr_z)
        for l in range(num_hidden):
            pred = effective_weights[l] @ relu(xs[l])
            eps = xs[l + 1] - pred
            early_eps_sq_sum[l] = early_eps_sq_sum[l] + eps ** 2

    remaining_steps = jnp.maximum(num_inference_steps - k_probe, 0)

    def body_fn(_, xs_inner):
        return _infer_step(
            effective_weights, xs_inner, precisions, lateral_pairs, lr_z
        )

    xs = jax.lax.fori_loop(0, remaining_steps, body_fn, xs)
    errors = compute_errors(effective_weights, xs)

    jacobians = compute_jacobians(effective_weights, xs)
    jacobians_per_sample = compute_jacobians_per_sample(effective_weights, xs)
    output_fisher = compute_output_fisher(effective_weights, xs)
    A_tildes = compute_local_maps(
        precisions,
        lateral_pairs,
        jacobians,
        ridge,
        output_fisher=output_fisher,
        jacobians_per_sample=jacobians_per_sample,
    )
    # Apply per-sample gates to per-sample Hebbian outer products, then
    # average across the batch. Diagnostics still use a 2D gate summary.
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
        delta_w = jnp.mean(gates_per_sample[l] * per_sample_hebbian, axis=0)
        gated_delta_w = vertical_eff_gates[l] * delta_w
        new_weights.append(
            weights[l] + lr_w * gated_delta_w - lambda_s * jnp.sign(weights[l])
        )

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

    new_effective_weights = apply_vertical_gates(
        new_weights, new_vertical_gate_logits, vertical_layer_scales
    )
    prediction = new_effective_weights[-1] @ relu(xs[-2])
    probs = jax.nn.softmax(prediction, axis=0)
    log_probs = jnp.log(probs + 1e-8)
    ce = -jnp.sum(y_batch.T * log_probs, axis=0)
    loss = jnp.mean(ce)

    return new_params, loss
