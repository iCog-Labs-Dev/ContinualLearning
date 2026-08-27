import jax
import jax.numpy as jnp

from generative.src.utils import get_activation, LayerParams, init_layer_params
from .energy_discriminative import compute_errors_discriminative


def compute_weight_gradients_discriminative(params, states, active_classes=None):
    """
    Analytical discriminative-PC weight gradients.

    W_i (params[i]) only appears in predicting x_{i+1} from x_i.
    """
    errors = compute_errors_discriminative(params, states, active_classes=active_classes)

    batch_size = states[0].shape[0]

    grads = []

    for i, layer in enumerate(params):
        lower_state = states[i]

        pre_activation = (
            lower_state @ layer["w"].T
            + layer["b"]
        )
        _, d_act_fn = get_activation(layer.get("activation", "tanh"))

        delta = (
            errors[i + 1]
            * d_act_fn(pre_activation)
        )

        dW = -(
            delta.T @ lower_state
        ) / batch_size

        db = -jnp.mean(
            delta,
            axis=0,
        )

        grads.append(
            {
                "w": dW,
                "b": db,
            }
        )

    return grads


def init_pcn_params_discriminative(key, layer_sizes, activation="tanh", use_bias=True):
    """
    Create all discriminative parameters.

    W_i maps x_i (dim layer_sizes[i]) -> x_{i+1} (dim layer_sizes[i+1]),
    i.e. fan_in = layer_sizes[i], fan_out = layer_sizes[i+1] — the REVERSE
    of the generative init's fan_in/fan_out assignment.
    """
    keys = jax.random.split(key, len(layer_sizes) - 1)

    params = []

    for k, (lower_dim, upper_dim) in zip(
        keys,
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        params.append(
            init_layer_params(
                k,
                fan_in=lower_dim,
                fan_out=upper_dim,
                activation=activation,
                use_bias=use_bias,
            )
        )

    return params


