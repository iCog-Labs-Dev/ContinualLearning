import jax
import jax.numpy as jnp

from .activations import relu, relu_derivative


def compute_errors(weights, xs):
    num_layers = len(xs)
    errors = [jnp.zeros_like(xs[0])]
    for l in range(1, num_layers):
        prediction = weights[l - 1] @ relu(xs[l - 1])
        if l == num_layers - 1:
            probs = jax.nn.softmax(prediction, axis=0)
            error = xs[l] - probs
        else:
            error = xs[l] - prediction
        errors.append(error)

    return errors


def _infer_step(weights, xs, precisions, laterals, lr_z):
    num_layers = len(xs)
    errors = compute_errors(weights, xs)
    new_xs = [xs[0]]
    for l in range(1, num_layers - 1):
        weighted_above = precisions[l][:, None] * errors[l + 1]
        weighted_self = precisions[l - 1][:, None] * errors[l]
        top_down = weights[l].T @ weighted_above * relu_derivative(xs[l])
        lateral = laterals[l - 1] @ xs[l]
        new_xs.append(xs[l] + lr_z * (top_down - weighted_self - lateral))

    new_xs.append(xs[-1])
    return new_xs


def infer(weights, xs, precisions, laterals, num_steps, lr_z):
    def body_fn(_, xs):
        return _infer_step(weights, xs, precisions, laterals, lr_z)

    return jax.lax.fori_loop(0, num_steps, body_fn, xs)
