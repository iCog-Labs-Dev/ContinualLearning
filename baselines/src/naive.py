import jax
import jax.numpy as jnp
from functools import partial
from core.model import MLP
from core.metrics import cross_entropy, accuracy
from core.data import Task


def _loss_fn(params, X, y, model: MLP):
    logits = model.forward(params, X)
    return cross_entropy(logits, y)


@partial(jax.jit, static_argnums=(4,))
def _train_step(params, X, y, lr, model):
    loss, grad = jax.value_and_grad(_loss_fn)(params, X, y, model)
    new_params = jax.tree.map(
        lambda params, gradiant: params - lr * gradiant, params, grad
    )

    return new_params, loss


class NaiveMethod:
    def __init__(self, lr, batch_size, epochs):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def train_task(self, model, params, task: Task):
        num_batch = task.train_X.shape[0] // self.batch_size

        for ep in range(self.epochs):
            total_loss = 0

            for i in range(num_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                batch_X = task.train_X[start:end]
                batch_y = task.train_y[start:end]

                params, loss = _train_step(params, batch_X, batch_y, self.lr, model)

                total_loss += loss

            print(f"Epoch: {ep + 1} ======== Loss: {total_loss / num_batch}")

        return params, total_loss / num_batch

    def evaluate(self, model: MLP, params, task: Task, allowed_classes=None):
        logits = model.forward(params, task.test_X)

        if allowed_classes is not None:
            mask = jnp.full((logits.shape[1],), -jnp.inf)
            mask = mask.at[jnp.array(allowed_classes)].set(0.0)
            logits = logits + mask

        predictions = jnp.argmax(logits, axis=1)
        return accuracy(predictions, task.test_y)
