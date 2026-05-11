import jax
import jax.numpy as jnp
from functools import partial

from core.metrics import cross_entropy
from core.model import MLP
from core.data import Task
from core.base import EWCState, EWCVanillaState
from .naive import _train_step


def _loss_fn(params, X, y, model: MLP):
    logits = model.forward(params, X)
    return cross_entropy(logits, y)


def _ewc_loss_fn(params, X, y, anchors, lam, model):
    task_loss = _loss_fn(params, X, y, model)
    total_penality = 0.0
    for anchor in anchors:
        penality = jax.tree.map(
            lambda F, p, p_old: jnp.sum(F * (p - p_old) ** 2),
            anchor["fisher"],
            params,
            anchor["params"],
        )
        total_penality = total_penality + sum(jax.tree.leaves(penality))
    return task_loss + (lam / 2) * total_penality


@partial(jax.jit, static_argnums=(6,))
def _ewc_train_step(params, X, y, anchors, lam, lr, model):
    loss, grad = jax.value_and_grad(_ewc_loss_fn)(params, X, y, anchors, lam, model)

    new_params = jax.tree.map(
        lambda params, gradiant: params - lr * gradiant, params, grad
    )

    return new_params, loss


class EWCMethod:
    def __init__(
        self,
        lr,
        lr_task1,
        batch_size,
        epochs,
        lam,
        num_samples,
        decay=1.0,
        anchor_alpha=0.0,
    ):
        self.lr = lr
        self.lr_task1 = lr_task1
        self.batch_size = batch_size
        self.epochs = epochs
        self.lam = lam
        self.num_samples = num_samples
        self.decay = decay
        self.anchor_alpha = anchor_alpha

    def compute_fisher(self, model: MLP, params, task: Task):
        X = task.train_X[: self.num_samples]
        y = task.train_y[: self.num_samples]

        def single_log_likelihood(params, x, y):
            logits = model.forward(params, x)
            log_probs = jax.nn.log_softmax(logits)
            return log_probs[y]

        grad_fn = jax.grad(single_log_likelihood)
        all_grad = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, X, y)
        fisher = jax.tree.map(lambda g: jnp.mean(g**2, axis=0), all_grad)
        return fisher

    def train_task(self, model: MLP, params, state, task: Task, task_idx):
        num_batch = task.train_X.shape[0] // self.batch_size

        for ep in range(self.epochs):
            total_loss = 0

            for i in range(num_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                batch_X = task.train_X[start:end]
                batch_y = task.train_y[start:end]

                if task_idx == 0:
                    params, loss = _train_step(
                        params, batch_X, batch_y, self.lr_task1, model
                    )
                else:
                    if self.decay < 1.0:
                        anchors = [
                            {
                                "fisher": state.cumulative_fisher,
                                "params": state.old_params,
                            }
                        ]
                    else:
                        anchors = state.anchors
                    params, loss = _ewc_train_step(
                        params,
                        batch_X,
                        batch_y,
                        anchors,
                        self.lam,
                        self.lr,
                        model,
                    )

                total_loss += loss

            print(f"Epoch: {ep + 1} Loss: {total_loss/num_batch}")

        new_fisher = self.compute_fisher(model, params, task)

        if self.decay < 1.0:
            if task_idx == 0:
                new_cumulative_fisher = new_fisher
            else:
                new_cumulative_fisher = jax.tree.map(
                    lambda cf, nf: self.decay * cf + nf,
                    state.cumulative_fisher,
                    new_fisher,
                )
            new_old_params = jax.tree.map(
                lambda old, new: self.anchor_alpha * old + (1 - self.anchor_alpha) * new,
                state.old_params,
                params,
            )
            return params, EWCState(old_params=new_old_params, cumulative_fisher=new_cumulative_fisher), total_loss / num_batch
        else:
            new_anchor = {"fisher": new_fisher, "params": params}
            return params, EWCVanillaState(anchors=state.anchors + [new_anchor]), total_loss / num_batch
