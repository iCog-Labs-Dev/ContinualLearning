

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

def relu(x):
    """ReLU activation."""
    return jnp.maximum(0.0, x)

def drelu(x):
    """Derivative of ReLU (subgradient: 0 at x=0)."""
    return (x > 0).astype(jnp.float32)

ACTIVATIONS = {
    "tanh":   (tanh,   dtanh),
    "linear": (linear, dlinear),
    "relu":   (relu,   drelu),
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
    Xavier/Glorot initialization — suited for bounded activations (tanh).

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


def he_init(key, fan_in, fan_out):
    """
    He/Kaiming initialization — suited for ReLU activations.

    std = sqrt(2 / fan_in)

    Returns:
        W: (fan_out, fan_in)
    """
    std = jnp.sqrt(2.0 / fan_in)
    return jax.random.normal(key, shape=(fan_out, fan_in)) * std


def init_layer_params(key, fan_in, fan_out, activation="tanh", use_bias=True):
    """
    Initialize one generative layer.

    Generative mapping:
        x_{i+1} -> x_i

    Weight shape:
        (fan_out, fan_in)

    Prediction:
        x_hat_i = act(x_{i+1} @ W.T + b)

    Parameters
    ----------
    activation:
        Name of activation function (``"tanh"``, ``"relu"``, ``"linear"``).
        He init is used automatically for ``"relu"``; Xavier for everything else.
    use_bias:
        If ``False``, bias is initialised to zeros and frozen during training.
    """
    w_key, _ = jax.random.split(key)

    # Use He init for ReLU, Xavier for everything else
    init_fn = he_init if activation == "relu" else xavier_init
    W = init_fn(w_key, fan_in, fan_out)

    b = jnp.zeros((fan_out,))

    return LayerParams({
        "w": W,
        "b": b,
        "activation": activation,
        "use_bias": use_bias,
    })



def init_pcn_params(key, layer_sizes, activation="tanh",
                    output_activation=None, use_bias=True):
    """
    Create all generative parameters.

    Parameters
    ----------
    activation:
        Activation used for all hidden generative layers (``params[1:]``).
    output_activation:
        Activation used for ``params[0]`` — the pixel-reconstruction layer
        that predicts the input from ``states[1]``. Defaults to ``activation``
        when ``None``. Set to ``"tanh"`` when using ``activation="relu"`` to
        keep pixel predictions bounded to ``[-1, 1]``.
    use_bias:
        Passed to every layer. Set ``False`` for paper-aligned ReLU runs.
    """
    if output_activation is None:
        output_activation = activation

    keys = jax.random.split(key, len(layer_sizes) - 1)

    params = []

    for idx, (k, (lower_dim, upper_dim)) in enumerate(zip(
        keys,
        zip(layer_sizes[:-1], layer_sizes[1:])
    )):
        # params[0] is the pixel-reconstruction layer — use output_activation
        layer_act = output_activation if idx == 0 else activation
        params.append(
            init_layer_params(
                k,
                fan_in=upper_dim,
                fan_out=lower_dim,
                activation=layer_act,
                use_bias=use_bias,
            )
        )

    return params