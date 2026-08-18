

import jax
import jax.numpy as jnp

from .utils import tanh, init_pcn_params
from .inference import settle_states
from .energy import compute_total_energy


class GenerativePCN:
    """
    Purely generative Predictive Coding Network.

    """

    def __init__(self, layer_sizes=None, activation="tanh",
                 output_activation=None, use_bias=True):
        if layer_sizes is None:
            layer_sizes = [784, 256, 256, 10]

        self.layer_sizes      = layer_sizes
        self.activation       = activation
        # output_activation applies to params[0] (pixel-reconstruction layer).
        # Defaults to `activation` when None.
        # When activation="relu", set output_activation="tanh" to keep pixel
        # predictions bounded to [-1, 1].
        self.output_activation = output_activation or activation
        self.use_bias         = use_bias

    def init_params(self, key):
        """
        Initialize all generative weights.

        Returns:
            params: list of layer parameter dictionaries
        """
        return init_pcn_params(
            key, self.layer_sizes,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
        )


    def forward(self, params, X, candidate_classes=jnp.arange(10), rescale=False, inference_steps=40):
        """Evaluate candidate classes and return negative energy as pseudo-logits.

        Parameters
        ----------
        params:
            Generative parameters (list of ``{"w", "b"}`` dicts).
        X:
            Input batch. Shape ``(batch_size, input_dim)``.
        candidate_classes:
            Which output classes to score. Defaults to all 10 MNIST digits.
        rescale:
            If ``True``, rescale ``X`` from ``[0, 1]`` to ``[-1, 1]`` before
            settling. Set this when ``X`` comes from the shared benchmark data
            pipeline (``core/data.py`` which normalises to ``[0, 1]``). Leave
            ``False`` when ``X`` is already in ``[-1, 1]`` (e.g. the standalone
            MNIST experiment).
        """
        if rescale:
            X = X * 2.0 - 1.0

        def evaluate_single_class(c):
            Y_c = jax.nn.one_hot(jnp.full((X.shape[0],), c), 10)
            states, _ = settle_states(params, X, Y_c, mode="bottom_up", num_steps=inference_steps)
            return -compute_total_energy(params, states)

        # vmap over classes
        pseudo_logits = jax.vmap(evaluate_single_class)(candidate_classes)
        return pseudo_logits.T  # Shape: (batch_size, num_classes)

    def forward_xl(self, params, X, rescale=False, inference_steps=40, eta_x=0.1):
        """Single-pass forward: clamp only X, let the top layer settle freely.

        Unlike ``forward()``, which runs full inference once per candidate
        class (cost = ``num_classes × B × T_infer``), this method clamps
        only the bottom layer and reads the **settled top-layer state**
        directly as class scores.

        Cost: ``1 × B × T_infer`` — independent of number of output classes.

        Parameters
        ----------
        params:
            Generative parameters (list of ``{"w", "b"}`` dicts).
        X:
            Input batch. Shape ``(batch_size, input_dim)``.
        rescale:
            If ``True``, rescale ``X`` from ``[0, 1]`` to ``[-1, 1]``
            before settling (use when X comes from ``core/data.py``).
        inference_steps:
            Number of settling steps (T_infer).
        eta_x:
            Inference learning rate.

        Returns
        -------
        logits : jnp.ndarray, shape ``(batch_size, num_classes)``
            Settled top-layer activations used directly as class scores.
        """
        if rescale:
            X = X * 2.0 - 1.0

        # Clamp only the input layer (index 0); top layer floats freely.
        # Y=None is safe here: init_states only reads Y when the top index
        # is in clamped_indices, which it isn't.
        states, _ = settle_states(
            params, X, None,
            num_steps=inference_steps,
            eta_x=eta_x,
            mode="bottom_up",
            clamped_indices=frozenset({0}),
        )

        # The settled state of the top layer is the inferred label vector.
        top_idx = len(params)   # == num_layers - 1
        return states[top_idx]  # shape (batch_size, num_classes)