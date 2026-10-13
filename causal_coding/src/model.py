import jax
import jax.numpy as jnp

from causal_coding.src.lateral import (
    adam_init_per_layer,
    inv_softplus,
)
from causal_coding.src.vertical_pruning import (
    apply_vertical_gates,
    init_vertical_gate_logits,
    init_vertical_importance,
    vertical_adam_init,
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
        lateral_mode="lowrank",
        struct_group_size=16,
        struct_alpha_e=0.13,
        struct_inhib_total=0.25,
        struct_rho=0.2,
    ):
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes)
        self.lateral_rank = lateral_rank
        self.lateral_init_scale = lateral_init_scale
        self.alpha_init = alpha_init
        # Structured lateral (paper's V). Single source of truth for rho / K.
        self.lateral_mode = lateral_mode
        self.struct_group_size = struct_group_size
        self.struct_alpha_e = struct_alpha_e
        self.struct_inhib_total = struct_inhib_total
        self.struct_rho = struct_rho

    def init_params(self, key):
        key_w, key_lU = jax.random.split(key, 2)
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

        vertical_gate_logits = init_vertical_gate_logits(weights, initial_gate=0.99)
        vertical_importance = init_vertical_importance(weights)
        vertical_adam_states = vertical_adam_init(vertical_gate_logits)
        vertical_layer_scales = [jnp.asarray(0.0) for _ in weights]
        vertical_adam_step = jnp.zeros((), dtype=jnp.float32)

        params = {
            "weights": weights,
            "log_precisions": log_precisions,
            "precision_var_ema": precision_var_ema,
            "lateral_U": lateral_U,
            "lateral_log_alpha": lateral_log_alpha,
            "lateral_cov_ema": lateral_cov_ema,
            "lateral_adam_states": lateral_adam_states,
            "lateral_adam_step": lateral_adam_step,
            "vertical_gate_logits": vertical_gate_logits,
            "vertical_importance": vertical_importance,
            "vertical_adam_states": vertical_adam_states,
            "vertical_adam_step": vertical_adam_step,
            "vertical_layer_scales": vertical_layer_scales,
            "vertical_match_loss": jnp.asarray(0.0),
            "vertical_sparse_loss": jnp.asarray(0.0),
        }
        if self.lateral_mode in ("structured", "both"):
            from causal_coding.src.lateral_ops import init_structured

            s, b, p_ema, q_ema = init_structured(
                hidden_dims, self.struct_group_size,
                self.struct_alpha_e, self.struct_inhib_total, self.struct_rho,
            )
            params.update({
                "struct_s": s,
                "struct_b": b,
                "struct_p_ema": p_ema,
                "struct_q_ema": q_ema,
                "struct_guard_frac": [jnp.zeros(()) for _ in hidden_dims],
            })
        return params

    def forward(self, params, X):
        h = X.T
        weights = params["weights"]
        if "vertical_gate_logits" in params and "vertical_layer_scales" in params:
            weights = apply_vertical_gates(
                weights,
                params["vertical_gate_logits"],
                params["vertical_layer_scales"],
            )
        for w in weights:
            h = w @ jnp.maximum(0.0, h)
        return h.T
