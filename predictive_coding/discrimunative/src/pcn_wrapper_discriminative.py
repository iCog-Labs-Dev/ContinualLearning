from __future__ import annotations

import jax.numpy as jnp

class PCNDiscriminativeModelWrapper:
    """Wraps DiscriminativePCN with rescale=True baked into forward()."""

    def __init__(self, pcn):
        self._pcn = pcn

    def __getattr__(self, name):
        return getattr(self._pcn, name)

    def forward(self, params, X, **kwargs):
        """Single-pass forward with automatic [0,1] -> [-1,1] rescaling."""
        return self._pcn.forward(params, X, rescale=True, **kwargs)
