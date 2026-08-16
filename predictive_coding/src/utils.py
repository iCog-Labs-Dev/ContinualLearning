

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


# Activation Registry

def linear(x):
    return x

def dlinear(x):
    return jnp.ones_like(x)

ACTIVATIONS = {
    "tanh": (tanh, dtanh),
    "linear": (linear, dlinear),
}

def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}")
    return ACTIVATIONS[name]


# Parameter Initialization

@jax.tree_util.register_pytree_node_class
class LayerParams(dict):
    """A dictionary that registers as a JAX PyTree, keeping 'w' and 'b' dynamic and other keys static."""
    def tree_flatten(self):
        dynamic = {k: v for k, v in self.items() if k in ("w", "b")}
        # Return tuple of items to make it hashable
        static = tuple(sorted((k, v) for k, v in self.items() if k not in ("w", "b")))
        return ((dynamic,), static)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        d = cls()
        d.update(children[0])
        d.update(dict(aux_data))
        return d

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


def init_layer_params(key, fan_in, fan_out, activation="tanh"):
    """
    Initialize one generative layer.

    Generative mapping:
        x_{i+1} -> x_i

    Weight shape:
        (fan_out, fan_in)

    Prediction:
        x_hat_i = act(x_{i+1} @ W.T + b)
    """
    w_key, b_key = jax.random.split(key)

    W = xavier_init(w_key, fan_in, fan_out)

    b = jnp.zeros((fan_out,))

    return LayerParams({
        "w": W,
        "b": b,
        "activation": activation,
    })



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