import jax
import jax.numpy as jnp
from functools import partial

from .activations import relu
from .inference import infer, compute_errors
from .do_influence import compute_jacobians, compute_local_maps, compute_causal_gates


@partial(jax.jit, static_argnums=(3,))
def train_step(
    params,
    x_batch,
    y_batch,
    num_inference_steps,
    lr_z,
    lr_w,
    lr_pi,
    lr_lat,
    gate_p,
    gate_kappa,
    ridge,
    lambda_s,
    rho_clarity,
):
    weights = params["weights"]
    log_precisions = params["log_precisions"]
    lateral_S_list = params["lateral_S"]

    num_layers = len(weights) + 1
    batch_size = x_batch.shape[0]

    precisions = [jnp.exp(lp) for lp in log_precisions]
    laterals = [s.T @ s for s in lateral_S_list]

    xs = [x_batch.T]
    for l in range(1, num_layers - 1):
        xs.append(weights[l - 1] @ relu(xs[l - 1]))
    xs.append(y_batch.T)

    xs = infer(weights, xs, precisions, laterals, num_inference_steps, lr_z)
    errors = compute_errors(weights, xs)

    jacobians = compute_jacobians(weights, xs)
    A_tildes = compute_local_maps(precisions, laterals, jacobians, ridge)
    gates = compute_causal_gates(A_tildes, gate_p, gate_kappa)

    new_weights = []
    for l in range(num_layers - 1):
        weighted_error = precisions[l][:, None] * errors[l + 1]
        delta_w = (1.0 / batch_size) * (weighted_error @ relu(xs[l]).T)
        gated_delta_w = gates[l] * delta_w
        new_weights.append(
            weights[l] + lr_w * gated_delta_w - lambda_s * jnp.sign(weights[l])
        )

    new_log_precisions = []
    for l in range(num_layers - 1):
        pi = precisions[l]
        mse = jnp.mean(errors[l + 1] ** 2, axis=1)
        dw = 0.5 * (1.0 - pi * mse)
        updated = log_precisions[l] + lr_pi * dw
        new_log_precisions.append(jnp.clip(updated, -4.0, 4.0))

    new_lateral_S = []
    num_lat = len(lateral_S_list)
    for l in range(num_lat):
        S = lateral_S_list[l]
        if l == num_lat - 1:

            new_lateral_S.append(S)
            continue
        x_eq = xs[l + 1]
        I_n = jnp.eye(S.shape[0])
        u = S @ x_eq
        Cu = (1.0 / batch_size) * (u @ u.T)
        eps_whiten = 1e-2
        Cu = Cu + eps_whiten * (S @ S.T)
        whiten = (I_n - Cu) @ S
        Lam = S.T @ S
        offdiag = jnp.sign(Lam) * (1.0 - I_n)
        clarity = 2.0 * (S @ offdiag)
        new_lateral_S.append(S + lr_lat * whiten - lr_lat * rho_clarity * clarity)

    new_params = {
        "weights": new_weights,
        "log_precisions": new_log_precisions,
        "lateral_S": new_lateral_S,
    }

    prediction = new_weights[-1] @ relu(xs[-2])
    probs = jax.nn.softmax(prediction, axis=0)
    log_probs = jnp.log(probs + 1e-8)
    ce = -jnp.sum(y_batch.T * log_probs, axis=0)
    loss = jnp.mean(ce)

    return new_params, loss
