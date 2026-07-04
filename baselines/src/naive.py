import jax
from functools import partial
from core.model import MLP
from core.metrics import compute_loss
from core.data import Task


def _loss_fn(params, X, y, model: MLP, active_classes=None):
    logits = model.forward(params, X)
    return compute_loss(logits, y, active_classes)


@partial(jax.jit, static_argnums=(4, 5))
def _train_step(params, X, y, lr, model, active_classes=None):
    loss, grad = jax.value_and_grad(_loss_fn)(params, X, y, model, active_classes)
    new_params = jax.tree.map(
        lambda params, gradiant: params - lr * gradiant, params, grad
    )

    return new_params, loss


class NaiveMethod:
    def __init__(self, lr, batch_size, epochs):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.task_il_training = False

    def train_task(self, model, params, state, task: Task, task_idx):
        num_batch = task.train_X.shape[0] // self.batch_size
        active_classes = tuple(task.classes) if self.task_il_training else None

        for ep in range(self.epochs):
            total_loss = 0

            for i in range(num_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                batch_X = task.train_X[start:end]
                batch_y = task.train_y[start:end]

                params, loss = _train_step(
                    params, batch_X, batch_y, self.lr, model, active_classes
                )

                total_loss += loss

            print(f"Epoch: {ep + 1} ======== Loss: {total_loss / num_batch}")

        return params, None, total_loss / num_batch
