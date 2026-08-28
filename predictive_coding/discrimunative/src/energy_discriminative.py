import jax
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
    e_i = x_i - x̂_i for hidden layers.
    For the final layer L, the error is determined by the classification loss:
    - Class-IL (active_classes is None): Softmax Cross-Entropy error (x_L - softmax(logits))
    - Task-IL (active_classes is not None): Sigmoid Binary Cross-Entropy error (x_L - sigmoid(logits)) masked to active classes.
    """
    predictions = predict_upper(params, states)
    errors = {}
    top_idx = len(params)

    # Hidden layer errors (Gaussian / L2)
    for i in range(1, top_idx):
        errors[i] = states[i] - predictions[i]

    # Output layer error
    logits = predictions[top_idx]
    targets = states[top_idx]

    if active_classes is None:
        errors[top_idx] = targets - jax.nn.softmax(logits, axis=-1)
    else:
        probs = jax.nn.sigmoid(logits)
        mask = jnp.zeros(logits.shape[-1]).at[jnp.array(active_classes)].set(1.0)
        errors[top_idx] = (targets - probs) * mask

    return errors


def compute_total_energy_discriminative(params, states, active_classes=None):
    """
    E = 1/2 Σ_{i=1}^{L-1} ||e_i||² + E_L
    where E_L is the classification loss (Cross-Entropy or BCE) at the final layer.
    """
    errors = compute_errors_discriminative(params, states, active_classes)
    top_idx = len(params)

    total_energy = 0.0
    for i in range(1, top_idx):
        total_energy += 0.5 * jnp.sum(jnp.square(errors[i]), axis=-1)

    # Classification energy for the output layer
    predictions = predict_upper(params, states)
    logits = predictions[top_idx]
    targets = states[top_idx]

    if active_classes is None:
        probs = jax.nn.softmax(logits, axis=-1)
        probs = jnp.clip(probs, 1e-15, 1.0 - 1e-15)
        output_energy = -jnp.sum(targets * jnp.log(probs), axis=-1)
    else:
        probs = jax.nn.sigmoid(logits)
        probs = jnp.clip(probs, 1e-15, 1.0 - 1e-15)
        bce = targets * jnp.log(probs) + (1.0 - targets) * jnp.log(1.0 - probs)
        mask = jnp.zeros(logits.shape[-1]).at[jnp.array(active_classes)].set(1.0)
        output_energy = -jnp.sum(bce * mask, axis=-1)

    total_energy += output_energy
    return total_energy


