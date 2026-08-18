
import jax.numpy as jnp


from .utils import get_activation


# predicts the lower layer by using  the upper layer states and the wieghts connecting them to it

def predict_lower(params, states):
    """
    Generic top-down predictions.

    params : list
       
        params[i] predicts state i from state i+1

    """

    predictions = {}

    for i, layer in enumerate(params):

        upper_state = states[i + 1]
        act_fn, _ = get_activation(layer.get("activation", "tanh"))

        predictions[i] = act_fn(
            upper_state @ layer["w"].T +
            layer["b"]
        )

    return predictions


# compute the errors which is the differene between the predicted state and the actual state 

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



# calculates the total energy which is simply the sum of all the errors 
def compute_total_energy(params, states):
    """
    E = 1/2 Σ ||e_i||²

    Returns per-sample energy of shape (batch_size,).
    """
    
    errors = compute_errors(params, states)

    total_energy = 0.0

    for e in errors.values():
        total_energy += 0.5 * jnp.sum(
            jnp.square(e),
            axis=-1
        )

    return total_energy