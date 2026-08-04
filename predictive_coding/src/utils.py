

import jax
import jax.numpy as jnp


# Activations

def tanh(x):
    """Analytical tanh activation."""
    return jnp.tanh(x)




def dtanh(x):
    """
    Analytical derivative of tanh.

    d/dx tanh(x) = 1 - tanh(x)^2
    """
    t = jnp.tanh(x)
    return 1.0 - t * t


# Parameter Initialization

def xavier_init(key, fan_in, fan_out):
    """
    Xavier/Glorot initialization.

    Returns:
        W: (fan_out, fan_in)
    """
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(
        key,
        shape=(fan_out, fan_in),
        minval=-limit,
        maxval=limit,
    )


def init_layer_params(key, fan_in, fan_out):
    """
    Initialize one generative layer.

    Generative mapping:
        x_{i+1} -> x_i

    Weight shape:
        (fan_out, fan_in)

    Prediction:
        x_hat_i = tanh(x_{i+1} @ W.T + b)
    """
    w_key, b_key = jax.random.split(key)

    W = xavier_init(w_key, fan_in, fan_out)

    b = jnp.zeros((fan_out,))

    return {
        "w": W,
        "b": b,
    }



def init_pcn_params(key, layer_sizes):
    """
    Create all generative parameters.
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
                fan_in=upper_dim,
                fan_out=lower_dim,
            )
        )

    return params