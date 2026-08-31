import jax
import jax.numpy as jnp

from .learning_discriminative import init_pcn_params_discriminative
from .inference_discriminative import settle_states_discriminative


class DiscriminativePCN:
    """
    Discriminative Predictive Coding Network.

    Weights predict UPWARD: params[i] maps x_i -> x̂_{i+1}.
    No generative/top-down pathway — cannot sample an image from a label.
    """

    def __init__(self, layer_sizes=None, activation="tanh", use_bias=True):
        if layer_sizes is None:
            layer_sizes = [784, 256, 256, 10]

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.use_bias = use_bias


    def init_params(self, key):
        return init_pcn_params_discriminative(
            key, self.layer_sizes,
            activation=self.activation,
            use_bias=self.use_bias,
        )

    def forward(self, params, X, active_classes=None, rescale=False, inference_steps=40, eta_x=0.1):
        """
        Single-pass inference: clamp X, free everything else, settle,
        read the top layer directly as class scores.

        """
        if rescale:
            X = X * 2.0 - 1.0

        if active_classes is not None:
            active_classes = tuple(active_classes)

        states, _ = settle_states_discriminative(
            params, X, None,
            num_steps=inference_steps,
            eta_x=eta_x,
            mode="bottom_up",
            clamped_indices=frozenset({0}),
            active_classes=active_classes,
        )

        top_idx = len(params)
        return states[top_idx]


    def forward_single_pass(self, params, X, rescale=False):
        """
        Deterministic single bottom-up pass, NO settling iterations.
        Sanity-check baseline: compare against `forward()` to confirm
        iterative settling is actually adding something.
        """
        from generative.src.utils import get_activation

        if rescale:
            X = X * 2.0 - 1.0

        current = X
        for layer in params:
            act_fn, _ = get_activation(layer.get("activation", "tanh"))
            current = act_fn(current @ layer["w"].T + layer["b"])

        return current
