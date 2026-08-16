

import jax
import jax.numpy as jnp

from .utils import tanh, init_pcn_params
from .inference import settle_states
from .energy import compute_total_energy


class GenerativePCN:
    """
    Purely generative Predictive Coding Network.

    """

    def __init__(self, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = [784, 256, 256, 10]

        self.layer_sizes = layer_sizes

    def init_params(self, key):
        """
        Initialize all generative weights.

        Returns:
            params: list of layer parameter dictionaries
        """
        return init_pcn_params(key, self.layer_sizes)

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