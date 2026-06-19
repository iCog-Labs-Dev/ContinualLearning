import jax
import jax.numpy as jnp
from core.model import MLP
from core.data import Task
from core.metrics import log_likelihood
from .ewc import EWCMethod


class EWCDRMethod(EWCMethod):
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
        super().__init__(
            lr, lr_task1, batch_size, epochs, lam, num_samples, decay, anchor_alpha
        )

    def compute_fisher(self, model: MLP, params, task: Task):
        X = task.train_X[: self.num_samples]
        y = task.train_y[: self.num_samples]
        active_classes = tuple(task.classes) if self.task_il_training else None

        def single_log_likelihood(params, x, y):
            logits = model.forward(params, x)
            if active_classes is not None:
                return log_likelihood(logits, y, active_classes)
            log_probs = jax.nn.log_softmax(-logits)
            return log_probs[y]

        grad_fn = jax.grad(single_log_likelihood)
        all_grad = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, X, y)
        fisher = jax.tree.map(lambda g: jnp.mean(g**2, axis=0), all_grad)
        epsilon = 1e-8
        fisher = jax.tree.map(lambda f: f / (jnp.max(f) + epsilon), fisher)
        return fisher
