import jax
import jax.numpy as jnp

from causal_coding.src.lateral import (
    adam_init_per_layer,
    inv_softplus,
)


def he_init(key, shape):
    fan_out, fan_in = shape
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / fan_in)


class CausalCodingModel:
    """
    Discriminative predictive-coding classifier with lateral state.

    The constructor takes lateral-init knobs but the rest of the lateral
    hyperparameters (rank, ε_lat, β_logdet, etc.) live on
    `CausalCodingMethod` — they're training-loop hyperparams, not model
    architecture.
    """

    def __init__(
        self,
        layer_sizes,
        lateral_rank=16,
        lateral_init_scale=1e-3,
        alpha_init=1e-4,
    ):
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes)
        self.lateral_rank = lateral_rank
        self.lateral_init_scale = lateral_init_scale
        self.alpha_init = alpha_init

    def init_params(self, key):
        key_w, key_lU = jax.random.split(key)
        weight_keys = jax.random.split(key_w, self.num_layers - 1)
        weights = [
            he_init(weight_keys[l], (self.layer_sizes[l + 1], self.layer_sizes[l]))
            for l in range(self.num_layers - 1)
        ]

        # Residual precision state. The output precision is kept fixed.
        log_precisions = [
            jnp.zeros(self.layer_sizes[l]) for l in range(1, self.num_layers)
        ]
        # Running hidden residual variance estimates for precision updates.
        precision_var_ema = [
            jnp.ones(self.layer_sizes[l]) for l in range(1, self.num_layers - 1)
        ]

        # Hidden-layer low-rank lateral precision state.
        num_hidden = self.num_layers - 2
        hidden_dims = [self.layer_sizes[l] for l in range(1, self.num_layers - 1)]

        if num_hidden > 0:
            U_keys = jax.random.split(key_lU, num_hidden)
            lateral_U = [
                self.lateral_init_scale
                * jax.random.normal(U_keys[i], (hidden_dims[i], self.lateral_rank))
                for i in range(num_hidden)
            ]
            # ρ init such that softplus(ρ) ≈ alpha_init.
            rho_init_value = inv_softplus(self.alpha_init)
            lateral_log_alpha = [jnp.asarray(rho_init_value) for _ in range(num_hidden)]

            # Start covariance estimates from an identity prior.
            lateral_cov_ema = [jnp.eye(d) for d in hidden_dims]
            lateral_adam_states = [
                adam_init_per_layer(U, rho)
                for U, rho in zip(lateral_U, lateral_log_alpha)
            ]
            lateral_adam_step = jnp.zeros((), dtype=jnp.float32)
        else:
            lateral_U = []
            lateral_log_alpha = []
            lateral_cov_ema = []
            lateral_adam_states = []
            lateral_adam_step = jnp.zeros((), dtype=jnp.float32)

        return {
            "weights": weights,
            "log_precisions": log_precisions,
            "precision_var_ema": precision_var_ema,
            "lateral_U": lateral_U,
            "lateral_log_alpha": lateral_log_alpha,
            "lateral_cov_ema": lateral_cov_ema,
            "lateral_adam_states": lateral_adam_states,
            "lateral_adam_step": lateral_adam_step,
        }

    def forward(self, params, X):
        h = X.T
        for w in params["weights"]:
            h = w @ jnp.maximum(0.0, h)
        return h.T
