import jax.numpy as jnp

from generative.src.utils import get_activation


# predicts the UPPER layer from the LOWER layer's state and the weights
# connecting them. params[i] predicts state i+1 from state i.

def predict_upper(params, states):
    """
    Discriminative (bottom-up) predictions.

    params[i] predicts state i+1 from state i.
    """
    predictions = {}

    for i, layer in enumerate(params):
        lower_state = states[i]
        act_fn, _ = get_activation(layer.get("activation", "tanh"))

        predictions[i + 1] = act_fn(
            lower_state @ layer["w"].T +
            layer["b"]
        )

    return predictions


def compute_errors_discriminative(params, states, active_classes=None):
    """
    e_{i+1} = x_{i+1} - x̂_{i+1}

    Keyed by the predicted (upper) state index, i.e. 1..L.
    """
    predictions = predict_upper(params, states)

    errors = {}
    for i in predictions:
        errors[i] = states[i] - predictions[i]

    if active_classes is not None:
        top_idx = len(params)
        mask = jnp.zeros(errors[top_idx].shape[-1])
        mask = mask.at[jnp.array(active_classes)].set(1.0)
        errors[top_idx] = errors[top_idx] * mask

    return errors


def compute_total_energy_discriminative(params, states, active_classes=None):
    """
    E = 1/2 Σ ||e_i||²

    Returns per-sample energy of shape (batch_size,).
    """
    errors = compute_errors_discriminative(params, states, active_classes)

    total_energy = 0.0
    for e in errors.values():
        total_energy += 0.5 * jnp.sum(jnp.square(e), axis=-1)

    return total_energy

