
import jax.numpy as jnp

from .utils import get_activation
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
        _, d_act_fn = get_activation(layer.get("activation", "tanh"))

        delta = (
            errors[i]
            * d_act_fn(pre_activation)
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
def update_weights(params, grads, eta_w=1e-3):
    """
    Gradient-descent parameter update.

    W <- W - eta_w * dW
    b <- b - eta_w * db
    """

    new_params = []

    for layer, grad in zip(params, grads):
        # Create a new dictionary or LayerParams preserving static keys
        new_layer = dict(layer)
        new_layer["w"] = layer["w"] - eta_w * grad["w"]

        # Freeze bias when use_bias=False (zeros, never updated)
        if layer.get("use_bias", True):
            new_layer["b"] = layer["b"] - eta_w * grad["b"]
        else:
            new_layer["b"] = layer["b"]

        # If it's a LayerParams, we need to return a LayerParams
        if type(layer).__name__ == "LayerParams":
            from .utils import LayerParams
            new_params.append(LayerParams(new_layer))
        else:
            new_params.append(new_layer)

    return new_params