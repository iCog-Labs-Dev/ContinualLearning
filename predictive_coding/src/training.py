import jax
import jax.numpy as jnp
from .inference import settle_states
from .learning import (
    compute_weight_gradients,
    update_weights,
)
from .energy import compute_total_energy
from functools import partial

@partial(jax.jit, static_argnames=("eta_x", "eta_w", "inference_steps", "init_mode"))
def train_step(
    params,
    X,
    Y,
    eta_x=0.1,
    eta_w=1e-3,
    inference_steps=50,
    init_mode="bottom_up",
):
    """
    One predictive-coding training step.

    Returns
    -------
    new_params
    metrics
    """

    states, energy_hist = settle_states(
        params,
        X,
        Y,
        num_steps=inference_steps,
        eta_x=eta_x,
        mode=init_mode,
    )

    grads = compute_weight_gradients(
        params,
        states,
    )

    new_params = update_weights(
        params,
        grads,
        eta_w=eta_w,
    )

    final_energy = jnp.mean(compute_total_energy(
        params,
        states,
    ))

    metrics = {
        "energy": final_energy,
        "final_inference_energy": energy_hist[-1],
        "energy_history": energy_hist,
    }

    return new_params, metrics