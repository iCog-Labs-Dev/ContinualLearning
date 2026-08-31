import jax
import jax.numpy as jnp
from functools import partial

from .inference_discriminative import settle_states_discriminative
from .learning_discriminative import compute_weight_gradients_discriminative
from generative.src.learning import update_weights


@partial(jax.jit, static_argnames=("eta_x", "eta_w", "inference_steps", "init_mode", "active_classes"))
def train_step_discriminative(
    params,
    X,
    Y,
    eta_x=0.1,
    eta_w=1e-3,
    inference_steps=50,
    init_mode="bottom_up",
    active_classes=None,
):
    """
    One discriminative-PC training step.

    Clamps {0, L} — input and label — exactly like generative training.

    Returns
    -------
    new_params
    metrics
    """
    num_states = len(params) + 1
    clamped_indices = frozenset({0, num_states - 1})

    states, energy_hist = settle_states_discriminative(
        params,
        X,
        Y,
        num_steps=inference_steps,
        eta_x=eta_x,
        mode=init_mode,
        clamped_indices=clamped_indices,
        active_classes=active_classes,
    )

    grads = compute_weight_gradients_discriminative(
        params,
        states,
        active_classes=active_classes,
    )

    new_params = update_weights(
        params,
        grads,
        eta_w=eta_w,
    )

    metrics = {
        "energy": energy_hist[-1],
        "energy_history": energy_hist,
    }

    return new_params, metrics
