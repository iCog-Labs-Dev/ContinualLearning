import jax
import jax.numpy as jnp
from functools import partial

from causal_coding.src.activations import relu
from causal_coding.src.inference import _infer_step, compute_errors
from causal_coding.src.do_influence import (
    compute_jacobians,
    compute_local_maps,
    compute_causal_gates,
    compute_output_fisher,
)
from causal_coding.src.lateral import (
    total_lateral_loss,
    adam_update_lateral,
    apply_spectral_cap,
    update_cov_ema,
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
    gate_alpha,
    gate_floor_coeff,
    beta_pi,
    epsilon_pi,
    alpha_pi,
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
):
    weights = params["weights"]
    log_precisions = params["log_precisions"]
    precision_var_ema = params["precision_var_ema"]
    lateral_U_list = params["lateral_U"]
    lateral_log_alpha_list = params["lateral_log_alpha"]
    lateral_cov_ema_list = params["lateral_cov_ema"]
    lateral_adam_states = params["lateral_adam_states"]
    lateral_adam_step = params["lateral_adam_step"]

    num_layers = len(weights) + 1
    batch_size = x_batch.shape[0]
    num_hidden = num_layers - 2  # excludes input (xs[0]) and output (xs[L])

    precisions = [jnp.exp(lp) for lp in log_precisions]

    # Build effective lateral_pairs (alpha_eff, U) for inference & Schur.
    raw_alphas = [jax.nn.softplus(rho) for rho in lateral_log_alpha_list]
    eff_alphas = [lateral_force_scale * a for a in raw_alphas]
    lateral_pairs = list(zip(eff_alphas, lateral_U_list))

    # Feedforward init + label clamp.
    xs = [x_batch.T]
    for l in range(1, num_layers - 1):
        xs.append(weights[l - 1] @ relu(xs[l - 1]))
    xs.append(y_batch.T)

    # Early inference steps with per-step residual capture for precision EMA.
    early_eps_sq_sum = [jnp.zeros_like(xs[l + 1]) for l in range(num_hidden)]
    for _ in range(k_probe):
        xs = _infer_step(weights, xs, precisions, lateral_pairs, lr_z)
        for l in range(num_hidden):
            pred = weights[l] @ relu(xs[l])
            eps = xs[l + 1] - pred
            early_eps_sq_sum[l] = early_eps_sq_sum[l] + eps ** 2

    # Continue remaining inference steps via fori_loop.
    def body_fn(_, xs_inner):
        return _infer_step(weights, xs_inner, precisions, lateral_pairs, lr_z)

    xs = jax.lax.fori_loop(0, num_inference_steps - k_probe, body_fn, xs)
    errors = compute_errors(weights, xs)

    jacobians = compute_jacobians(weights, xs)
    output_fisher = compute_output_fisher(weights, xs)
    A_tildes = compute_local_maps(
        precisions, lateral_pairs, jacobians, ridge, output_fisher=output_fisher
    )
    gates = compute_causal_gates(A_tildes, gate_p, gate_kappa, gate_alpha, gate_floor_coeff)

    # Vertical weight update: local Hebbian error update, causal gate, and L1 shrinkage.
    new_weights = []
    for l in range(num_layers - 1):
        weighted_error = precisions[l][:, None] * errors[l + 1]
        delta_w = (1.0 / batch_size) * (weighted_error @ relu(xs[l]).T)
        gated_delta_w = gates[l] * delta_w
        new_weights.append(
            weights[l] + lr_w * gated_delta_w - lambda_s * jnp.sign(weights[l])
        )

    # Update hidden-layer residual precision from early-inference residual variance.
    new_precision_var_ema = []
    new_log_precisions = []
    for l in range(num_hidden):
        mean_batch_eps_sq = jnp.mean(early_eps_sq_sum[l] / k_probe, axis=1)
        new_v = beta_pi * precision_var_ema[l] + (1.0 - beta_pi) * mean_batch_eps_sq
        target_log_pi = jnp.clip(-jnp.log(new_v + epsilon_pi), -4.0, 4.0)
        damped_log_pi = (1.0 - alpha_pi) * log_precisions[l] + alpha_pi * target_log_pi
        new_precision_var_ema.append(new_v)
        new_log_precisions.append(damped_log_pi)
    new_log_precisions.append(log_precisions[-1])  # output frozen

    # Update hidden-state covariance estimates and lateral precision parameters.
    new_cov_emas = []
    for l_idx in range(num_hidden):
        z_l_eq = xs[l_idx + 1]  # xs[1], xs[2] for two hidden layers
        new_cov_emas.append(
            update_cov_ema(lateral_cov_ema_list[l_idx], z_l_eq, beta_cov)
        )

    # Stop gradients through the state covariance; lateral learning should not
    # backpropagate through the predictive-coding inference trajectory.
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

    # Bound the maximum possible lateral strength, independent of the ramp
    # currently applied during inference.
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
    }

    prediction = new_weights[-1] @ relu(xs[-2])
    probs = jax.nn.softmax(prediction, axis=0)
    log_probs = jnp.log(probs + 1e-8)
    ce = -jnp.sum(y_batch.T * log_probs, axis=0)
    loss = jnp.mean(ce)

    return new_params, loss
