import jax
import jax.numpy as jnp


def he_init(key, shape):
    fan_out, fan_in = shape
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / fan_in)


class CausalCodingModel:
    def __init__(self, layer_sizes, lateral_init_scale=0.01):
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes)
        self.lateral_init_scale = lateral_init_scale

    def init_params(self, key):
        key_w, key_l = jax.random.split(key)
        weight_keys = jax.random.split(key_w, self.num_layers - 1)
        weights = [
            he_init(weight_keys[l], (self.layer_sizes[l + 1], self.layer_sizes[l]))
            for l in range(self.num_layers - 1)
        ]

        log_precisions = [
            jnp.zeros(self.layer_sizes[l]) for l in range(1, self.num_layers)
        ]

        num_laterals = self.num_layers - 1

        if num_laterals > 0:
            lateral_keys = jax.random.split(key_l, num_laterals)
            lateral_S = [
                self.lateral_init_scale
                * jax.random.normal(
                    lateral_keys[l],
                    (self.layer_sizes[l + 1], self.layer_sizes[l + 1]),
                )
                for l in range(num_laterals)
            ]
        else:
            lateral_S = []

        return {
            "weights": weights,
            "log_precisions": log_precisions,
            "lateral_S": lateral_S,
        }

    def forward(self, params, X):
        h = X.T
        for w in params["weights"]:
            h = w @ jnp.maximum(0.0, h)
        return h.T
