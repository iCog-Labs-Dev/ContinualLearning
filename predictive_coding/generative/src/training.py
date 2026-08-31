import jax
import jax.numpy as jnp
import optax
from .inference import settle_states
from .learning import (
    compute_weight_gradients,
    update_weights,
)
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

    metrics = {
        "energy": energy_hist[-1],
        "final_inference_energy": energy_hist[-1],
        "energy_history": energy_hist,
    }

    return new_params, metrics


@partial(jax.jit, static_argnames=("eta_x", "inference_steps", "init_mode", "optimizer"))
def train_step_optax(
    params,
    opt_state,
    X,
    Y,
    optimizer,
    eta_x=0.1,
    inference_steps=50,
    init_mode="bottom_up",
):
    """
    One predictive-coding training step using optax for optimization.

    Returns
    -------
    new_params
    new_opt_state
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

    # Optax expects a list of dictionaries with matching structure to params.
    # However, if use_bias=False, we want to freeze the bias by zeroing out the gradient.
    masked_grads = []
    from .utils import LayerParams
    for p, g in zip(params, grads):
        if not p.get("use_bias", True):
            db = jnp.zeros_like(g["b"])
        else:
            db = g["b"]
            
        if type(p).__name__ == "LayerParams":
            new_g = dict(p)
            new_g["w"] = g["w"]
            new_g["b"] = db
            masked_grads.append(LayerParams(new_g))
        else:
            masked_grads.append({"w": g["w"], "b": db})
            
    updates, new_opt_state = optimizer.update(masked_grads, opt_state, params)
    
    # We must explicitly cast params to a list for optax if it involves custom PyTrees 
    # but since LayerParams is registered, it works transparently.
    new_params = optax.apply_updates(params, updates)

    metrics = {
        "energy": energy_hist[-1],
        "final_inference_energy": energy_hist[-1],
        "energy_history": energy_hist,
    }

    return new_params, new_opt_state, metrics