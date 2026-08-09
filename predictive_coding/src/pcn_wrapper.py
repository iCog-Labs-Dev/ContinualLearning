"""Thin wrapper that bakes rescale=True into GenerativePCN.forward().
The benchmark ``Evaluator`` calls::

    logits = model.forward(params, task.test_X)
``task.test_X`` comes from ``core/data.py`` normalised to ``[0, 1]``. The PCN needs ``[-1, 1]``.

so  we made t
``forward(params, X)`` automatically applies ``X = X * 2 - 1`` 
"""

from __future__ import annotations

import jax.numpy as jnp




class PCNModelWrapper:
    """Wraps ``GenerativePCN`` with ``rescale=True`` baked into ``forward()``.

    """

    def __init__(self, pcn):
        self._pcn = pcn

    # Delegate attribute access to the underlying model 
    def __getattr__(self, name):
        return getattr(self._pcn, name)

    def forward(self, params, X, candidate_classes=jnp.arange(10)):
        """Forward pass with automatic ``[0,1] -> [-1,1]`` rescaling."""
        return self._pcn.forward(
            params, X, candidate_classes=candidate_classes, rescale=True
        )
