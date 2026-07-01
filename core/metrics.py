import jax
import jax.numpy as jnp


def _binary_cross_entropy(logits, targets):
    return (
        jnp.maximum(logits, 0)
        - logits * targets
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


def cross_entropy(logits, labels):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def task_il_bce(logits, labels, active_classes):
    active = jnp.array(active_classes)
    task_logits = logits[:, active]
    targets = (labels[:, None] == active[None, :]).astype(jnp.float32)
    return jnp.mean(_binary_cross_entropy(task_logits, targets))


def compute_loss(logits, labels, active_classes=None):
    if active_classes is None:
        return cross_entropy(logits, labels)
    return task_il_bce(logits, labels, active_classes)


def log_likelihood(logits, label, active_classes=None):
    if active_classes is None:
        log_probs = jax.nn.log_softmax(logits)
        return log_probs[label]
    active = jnp.array(active_classes)
    task_logits = logits[active]
    targets = (active == label).astype(jnp.float32)
    bce = _binary_cross_entropy(task_logits, targets)
    return -jnp.sum(bce)


def class_il_predict(logits):
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.argmax(probs, axis=1)


def task_il_predict(logits, active_classes):
    active = jnp.array(active_classes)
    task_logits = logits[:, active]
    probs = jax.nn.sigmoid(task_logits)
    local_pred = jnp.argmax(probs, axis=1)
    return active[local_pred]


def accuracy(predictions, labels):
    return jnp.mean(predictions == labels)
