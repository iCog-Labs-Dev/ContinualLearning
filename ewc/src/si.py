import jax
import jax.numpy as jnp
from functools import partial
from .model import MLP
from .utils import cross_entropy, accuracy
from .data import Task


def _loss_fn(params, X, y, model: MLP):
    logits = model.forward(params, X)
    return cross_entropy(logits, y)


def _si_loss_fn(params, X, y, old_params, omega, lam, model: MLP):
    task_loss = _loss_fn(params, X, y, model)
    penality = jax.tree.map(
        lambda o, p, op: jnp.sum(o * (p - op) ** 2), omega, params, old_params
    )

    total_penality = sum(jax.tree.leaves(penality))
    return task_loss + (lam / 2) * total_penality


@partial(jax.jit, static_argnums=(8,))
def _si_train_step(params, X, y, old_params, omega, lam, lr, contribution_sum, model):
    (loss, grad) = jax.value_and_grad(_si_loss_fn)(
        params, X, y, old_params, omega, lam, model
    )

    new_params = jax.tree.map(lambda params, grad: params - lr * grad, params, grad)

    delta = jax.tree.map(lambda np, p: np - p, new_params, params)
    new_contribution = jax.tree.map(
        lambda c_s, delta, grad: c_s + (-grad * delta), contribution_sum, delta, grad
    )

    return new_params, loss, new_contribution


class SIMethod:
    def __init__(
        self,
        lr,
        lr_task1,
        batch_size,
        epochs,
        lam,
        epsilon=1e-3,
        decay=1.0,
        normalize=False,
    ):
        self.lr = lr
        self.lr_task1 = lr_task1
        self.batch_size = batch_size
        self.epochs = epochs
        self.lam = lam
        self.epsilon = epsilon
        self.decay = decay
        self.normalize = normalize

    def train_task(self, model, params, state, task: Task, task_idx):
        initial_params = params
        contribution_sum = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        old_params = state["old_params"]
        cumulative_omega = state["cumulative_omega"]

        for ep in range(self.epochs):
            num_batch = task.train_X.shape[0] // self.batch_size
            total_loss = 0

            for i in range(num_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size

                batch_X = task.train_X[start:end]
                batch_y = task.train_y[start:end]

                if task_idx == 0:
                    new_params, loss, new_contribution = _si_train_step(
                        params,
                        batch_X,
                        batch_y,
                        old_params,
                        cumulative_omega,
                        self.lam,
                        self.lr_task1,
                        contribution_sum,
                        model,
                    )
                else:
                    new_params, loss, new_contribution = _si_train_step(
                        params,
                        batch_X,
                        batch_y,
                        old_params,
                        cumulative_omega,
                        self.lam,
                        self.lr,
                        contribution_sum,
                        model,
                    )

                params = new_params
                contribution_sum = new_contribution
                total_loss += loss

            print(f"Epoch {ep + 1}: Loss {total_loss / num_batch}")

        total_distance = jax.tree.map(lambda p, ip: p - ip, params, initial_params)
        omega_new = jax.tree.map(
            lambda contrib, dist: jnp.maximum(0, contrib / (dist**2 + self.epsilon)),
            contribution_sum,
            total_distance,
        )
        if self.normalize:
            omega_new = jax.tree.map(
                lambda o: o / (jnp.max(o) + self.epsilon), omega_new
            )

        if task_idx == 0:
            new_cumulative_omega = omega_new
        else:
            new_cumulative_omega = jax.tree.map(
                lambda c_o, o_n: self.decay * c_o + o_n, cumulative_omega, omega_new
            )

        new_state = {"old_params": params, "cumulative_omega": new_cumulative_omega}
        return params, new_state, total_loss / num_batch

    def evaluate(self, model: MLP, params, task: Task):
        logits = model.forward(params, task.test_X)
        predictions = jnp.argmax(logits, axis=1)
        return accuracy(predictions, task.test_y)
