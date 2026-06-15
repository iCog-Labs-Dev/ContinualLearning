import jax.numpy as jnp


def relu(x):
    return jnp.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0).astype(x.dtype)
