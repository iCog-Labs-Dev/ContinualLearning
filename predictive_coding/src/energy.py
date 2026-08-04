
import jax.numpy as jnp


from .utils import tanh


def predict_lower(params, states):
    """
    Generic top-down predictions.

    Parameters
    ----------
    params : list
       
        params[i] predicts state i from state i+1

    Returns
    -------
    """

    predictions = {}

    for i, layer in enumerate(params):

        upper_state = states[i + 1]

        predictions[i] = tanh(
            upper_state @ layer["w"].T +
            layer["b"]
        )

    return predictions


def compute_errors(params, states):
    """
    Compute all prediction errors.

    e_i = x_i - x̂_i
    """

    predictions = predict_lower(params, states)

    errors = {}

    for i in predictions:
        errors[i] = states[i] - predictions[i]

    return errors




def compute_total_energy(params, states):
    """
    E = 1/2 Σ ||e_i||²

    Returns scalar batch-mean energy.
    """

    errors = compute_errors(params, states)

    total_energy = 0.0

    for e in errors.values():
        total_energy += 0.5 * jnp.sum(
            jnp.square(e),
            axis=-1
        )

    return jnp.mean(total_energy)