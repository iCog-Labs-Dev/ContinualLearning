
import jax.numpy as jnp
from .utils import tanh
from .energy import predict_lower



def init_states(params, X, Y, mode="zero"):
    """
    Create the latent states used during inference.

    States convention:
        x0 = input image (fixed)
        xL = label (fixed)

    Hidden states:
        x1 ... x(L-1)
    """

    states = {}

    states[0] = X

    num_layers = len(params) + 1

    batch_size = X.shape[0]

    if mode == "zero":

        for i in range(1, num_layers - 1):

            hidden_dim = params[i - 1]["w"].shape[1]

            states[i] = jnp.zeros(
                (batch_size, hidden_dim)
            )

    elif mode == "bottom_up":

        states[num_layers - 1] = Y

        predictions = predict_lower(params, states)

        for i in reversed(range(1, num_layers - 1)):
            states[i] = predictions[i]

    else:
        raise ValueError(
            f"Unknown init mode: {mode}"
        )

    states[num_layers - 1] = Y

    return states