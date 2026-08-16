import jax
import jax.numpy as jnp

from .utils import get_activation
from .energy import compute_errors, compute_total_energy

from functools import partial


# initialize the states at each layer 
def init_states(params, X, Y, mode="zero", clamped_indices=None):
    num_states = len(params) + 1
    if clamped_indices is None:
        clamped_indices = frozenset({0, num_states - 1})

    batch_size = X.shape[0]
    states = {}
    
    if 0 in clamped_indices:
        states[0] = X
    if (num_states - 1) in clamped_indices:
        states[num_states - 1] = Y

    if mode == "zero":
        for i in range(num_states):
            if i not in clamped_indices:
                if i == num_states - 1:
                    states[i] = jnp.zeros_like(Y) if Y is not None else jnp.zeros((batch_size, params[-1]["w"].shape[1]))
                elif i == 0:
                    states[i] = jnp.zeros_like(X)
                else:
                    hidden_dim = params[i - 1]["w"].shape[1]
                    states[i] = jnp.zeros((batch_size, hidden_dim))

    elif mode == "bottom_up":
        current = X if 0 in clamped_indices else jnp.zeros_like(X)
        if 0 not in clamped_indices:
            states[0] = current

        for i in range(num_states - 1):
            W = params[i]["w"]
            act_fn, _ = get_activation(params[i].get("activation", "tanh"))
            current = act_fn(current @ W)
            if (i + 1) not in clamped_indices:
                states[i + 1] = current

    return states


# computes the gradient of the energy with respect to the state of each node 
def compute_state_gradients(params, states, clamped_indices=None):
    """
    Analytical predictive-coding state gradients.
    """
    num_states = len(params) + 1
    if clamped_indices is None:
        clamped_indices = frozenset({0, num_states - 1})

    errors = compute_errors(params, states)
    grads = {}

    for i in range(num_states):
        if i in clamped_indices:
            continue

        grad = errors[i] if i in errors else 0.0

        if i > 0:
            W_lower = params[i - 1]["w"]
            b_lower = params[i - 1]["b"]

            pre_activation = (
                states[i] @ W_lower.T
                + b_lower
            )
            _, d_act_fn = get_activation(params[i - 1].get("activation", "tanh"))

            feedback = (
                errors[i - 1]
                * d_act_fn(pre_activation)
            ) @ W_lower

            grad = grad - feedback

        grads[i] = grad

    return grads


# runs the process of setling the states 
@partial(
    jax.jit,
    static_argnames=("num_steps", "eta_x", "mode", "clamped_indices")
)
def settle_states(params, X, Y, num_steps=50, eta_x=0.1, mode="zero", clamped_indices=None):
    """
    Iterative predictive-coding inference.
    """
    num_states = len(params) + 1
    if clamped_indices is None:
        clamped_indices = frozenset({0, num_states - 1})

    states = init_states(params, X, Y, mode=mode, clamped_indices=clamped_indices)
    energy_hist = jnp.zeros((num_steps,))

    def step_fn(step_idx, carry):
        states, energy_hist = carry
        grads = compute_state_gradients(params, states, clamped_indices=clamped_indices)
    
        new_states = dict(states)
        for i in range(num_states):
            if i not in clamped_indices:
                new_states[i] = states[i] - eta_x * grads[i]

        energy = jnp.mean(compute_total_energy(params, new_states))
        energy_hist = energy_hist.at[step_idx].set(energy)
        return new_states, energy_hist

    states, energy_hist = jax.lax.fori_loop(0, num_steps, step_fn, (states, energy_hist))
    return states, energy_hist


