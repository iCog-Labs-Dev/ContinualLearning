
import jax.numpy as jnp

from .utils import tanh ,dtanh
from .energy import compute_errors , compute_total_energy

from functools import partial



def init_states(params, X, Y, mode="zero"):
    """
    Initialize latent states for predictive-coding inference.

    """

    batch_size = X.shape[0]

    states = {}
    states[0] = X

    num_states = len(params) + 1

    if mode == "zero":

        for i in range(1, num_states):

            hidden_dim = params[i - 1]["w"].shape[1]

            states[i] = jnp.zeros(
                (batch_size, hidden_dim)
            )

    elif mode == "bottom_up":

        current = X

        for i in range(num_states - 1):

            W = params[i]["w"]

            current = tanh(current @ W)

            states[i + 1] = current

    else:
        raise ValueError(
            f"Unknown initialization mode: {mode}"
        )


    states[num_states] = Y

    return states






def compute_state_gradients(params, states):
    """
    Analytical predictive-coding state gradients.
    """

    errors = compute_errors(params, states)

    grads = {}

    num_states = len(params) + 1

    for i in range(1, num_states - 1):

        grad = errors[i]

        W_lower = params[i - 1]["w"]
        b_lower = params[i - 1]["b"]

        pre_activation = (
            states[i] @ W_lower.T
            + b_lower
        )

        feedback = (
            errors[i - 1]
            * dtanh(pre_activation)
        ) @ W_lower

        grad = grad - feedback

        grads[i] = grad

    return grads








@partial(
    jax.jit,
    static_argnames=("num_steps", "eta_x", "mode")
)
def settle_states (params, X, Y, num_steps=50, eta_x=0.1, mode="zero"):

    """
    Iterative predictive-coding inference.
    Hidden states are updated via:

        x_i <- x_i - eta_x * dE/dx_i

    Input and label layers remain clamped.

    Returns
    -------
    states
        Settled states.

    energy_hist
        Energy at every inference step.
    """

    states = init_states(
        params,
        X,
        Y,
        mode=mode,
    )

    energy_hist = jnp.zeros((num_steps,))

    num_states = len(params) + 1

    def step_fn(step_idx, carry):

        states, energy_hist = carry

        grads = compute_state_gradients(
            params,
            states,
        )
    
        # update latent states only
        new_states = dict(states)

        for i in range(1, num_states - 1):

            new_states[i] = (
                states[i]
                - eta_x * grads[i]
            )

        energy = compute_total_energy(
            params,
            new_states,
        )

        energy_hist = energy_hist.at[
            step_idx
        ].set(energy)

        return new_states, energy_hist

    states, energy_hist = jax.lax.fori_loop(
        0,
        num_steps,
        step_fn,
        (
            states,
            energy_hist,
        ),
    )

    return states, energy_hist


