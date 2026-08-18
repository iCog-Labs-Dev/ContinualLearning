import jax
import jax.numpy as jnp

from generative.src.utils import get_activation
from .energy_discriminative import (
    compute_errors_discriminative,
    compute_total_energy_discriminative,
)

from functools import partial


def init_states_discriminative(params, X, Y, mode="zero", clamped_indices=None):
    """
    Same contract as generative init_states, but bottom-up mode propagates
    forward through predict_upper (params[i] maps state i -> state i+1).
    """
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
                    states[i] = (
                        jnp.zeros_like(Y) if Y is not None
                        else jnp.zeros((batch_size, params[-1]["w"].shape[0]))
                    )
                elif i == 0:
                    states[i] = jnp.zeros_like(X)
                else:
                    hidden_dim = params[i - 1]["w"].shape[0]  # fan_out of params[i-1]
                    states[i] = jnp.zeros((batch_size, hidden_dim))

    elif mode == "bottom_up":
        current = X if 0 in clamped_indices else jnp.zeros_like(X)
        if 0 not in clamped_indices:
            states[0] = current

        for i in range(num_states - 1):
            W = params[i]["w"]
            b = params[i]["b"]
            act_fn, _ = get_activation(params[i].get("activation", "tanh"))
            current = act_fn(current @ W.T + b)
            if (i + 1) not in clamped_indices:
                states[i + 1] = current

    return states


def compute_state_gradients_discriminative(params, states, clamped_indices=None):
    """
    Analytical discriminative-PC state gradients.

    For hidden state x_i:
      - local term: its own prediction error, errors[i] (predicted by params[i-1])
      - feedback term: pulled by errors[i+1], propagated back through params[i]
        (the layer that uses x_i as input to predict x_{i+1})
    """
    num_states = len(params) + 1
    if clamped_indices is None:
        clamped_indices = frozenset({0, num_states - 1})

    errors = compute_errors_discriminative(params, states)
    grads = {}

    for i in range(num_states):
        if i in clamped_indices:
            continue

        grad = errors[i] if i in errors else 0.0

        if i < len(params):  # params[i] exists: x_i is used to predict x_{i+1}
            W_i = params[i]["w"]
            b_i = params[i]["b"]

            pre_activation = states[i] @ W_i.T + b_i
            _, d_act_fn = get_activation(params[i].get("activation", "tanh"))

            feedback = (
                errors[i + 1] * d_act_fn(pre_activation)
            ) @ W_i

            grad = grad - feedback

        grads[i] = grad

    return grads


@partial(
    jax.jit,
    static_argnames=("num_steps", "eta_x", "mode", "clamped_indices")
)
def settle_states_discriminative(
    params, X, Y, num_steps=50, eta_x=0.1, mode="zero", clamped_indices=None
):
    """
    Iterative discriminative-PC inference.
    """
    num_states = len(params) + 1
    if clamped_indices is None:
        clamped_indices = frozenset({0, num_states - 1})

    states = init_states_discriminative(
        params, X, Y, mode=mode, clamped_indices=clamped_indices
    )
    energy_hist = jnp.zeros((num_steps,))

    def step_fn(step_idx, carry):
        states, energy_hist = carry
        grads = compute_state_gradients_discriminative(
            params, states, clamped_indices=clamped_indices
        )

        new_states = dict(states)
        for i in range(num_states):
            if i not in clamped_indices:
                new_states[i] = states[i] - eta_x * grads[i]

        energy = jnp.mean(compute_total_energy_discriminative(params, new_states))
        energy_hist = energy_hist.at[step_idx].set(energy)
        return new_states, energy_hist

    states, energy_hist = jax.lax.fori_loop(0, num_steps, step_fn, (states, energy_hist))
    return states, energy_hist
