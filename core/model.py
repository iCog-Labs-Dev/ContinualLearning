import jax
import jax.numpy as jnp


def he_init(key, fan_in, fan_out):
    return jax.random.normal(key, shape=(fan_in, fan_out)) * jnp.sqrt(2 / fan_in)


class MLP:
    def __init__(self, layer_sizes: list):
        self.layer_sizes = layer_sizes

    def init_params(self, key):
        params = {}

        for i in range(len(self.layer_sizes) - 1):
            key, sub_key = jax.random.split(key)
            params[f"layer_{i + 1}"] = {
                "w": he_init(sub_key, self.layer_sizes[i], self.layer_sizes[i + 1]),
                "b": jnp.zeros(self.layer_sizes[i + 1]),
            }

        return params

    def forward(self, params, X):
        num_layers = len(params)
        z = 0.0
        for i in range(num_layers):
            W = params[f"layer_{i + 1}"]["w"]
            b = params[f"layer_{i + 1}"]["b"]

            if i < num_layers - 1:
                z = X @ W + b
                X = jnp.maximum(0, z)
            else:
                z = X @ W + b

        return z

    def forward_with_states(self, params, X):
        num_layers = len(params)
        states = []
        z = 0.0
        for i in range(num_layers):
            W = params[f"layer_{i + 1}"]["w"]
            b = params[f"layer_{i + 1}"]["b"]

            if i < num_layers - 1:
                z = X @ W + b
                X = jnp.maximum(0, z)
                states.append(X)
            else:
                z = X @ W + b
                states.append(z)

        return states, z
