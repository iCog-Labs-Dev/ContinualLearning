
import jax.numpy as jnp

from .utils import dtanh
from .energy import compute_errors



# compute the gradients of the energy with respect to the weights and biases 

def compute_weight_gradients(params, states):
    """
    Analytical predictive-coding weight gradients.
    """

    errors = compute_errors(
        params,
        states,
    )

    batch_size = states[0].shape[0]

    grads = []

    for i, layer in enumerate(params):

        upper_state = states[i + 1]

        pre_activation = (
            upper_state @ layer["w"].T
            + layer["b"]
        )

        delta = (
            errors[i]
            * dtanh(pre_activation)
        )

        dW = -(
            delta.T @ upper_state
        ) / batch_size

        db = -jnp.mean(
            delta,
            axis=0,
        )

        grads.append(
            {
                "w": dW,
                "b": db,
            }
        )

    return grads



# update the weights using gradient descent 
def update_weights(
    params,
    grads,
    eta_w=1e-3,
):
    """
    Gradient-descent parameter update.

    W <- W - eta_w * dW
    b <- b - eta_w * db
    """

    new_params = []

    for layer, grad in zip(params, grads):

        new_params.append(
            {
                "w": layer["w"] - eta_w * grad["w"],
                "b": layer["b"] - eta_w * grad["b"],
            }
        )

    return new_params