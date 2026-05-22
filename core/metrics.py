import jax
import jax.numpy as jnp


def cross_entropy(logits, labels):
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def accuracy(predictions, labels):
    return jnp.mean(predictions == labels)
