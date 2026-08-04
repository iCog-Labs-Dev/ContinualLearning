

import jax
import jax.numpy as jnp

from .utils import tanh, init_pcn_params



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

    def forward(self, params, X, candidate_classes=None):
        """
        Placeholder.

        """
        batch_size = X.shape[0]

        return jnp.zeros((batch_size, 10))


    def forward(self, params, X, candidate_classes=None):
        batch_size = X.shape[0]
        return jnp.zeros((batch_size, 10))