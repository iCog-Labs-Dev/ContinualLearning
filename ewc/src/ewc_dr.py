import jax
import jax.numpy as jnp
from .model import MLP
from .data import Task
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
        decay,
        anchor_alpha=0.0,
    ):
        super().__init__(
            lr, lr_task1, batch_size, epochs, lam, num_samples, decay, anchor_alpha
        )

    def compute_fisher(self, model: MLP, params, task: Task):
        X = task.train_X[: self.num_samples]
        y = task.train_y[: self.num_samples]

        def single_log_likelihood(params, x, y):
            logits = model.forward(params, x)
            log_probs = jax.nn.log_softmax(-logits)
            return log_probs[y]

        grad_fn = jax.grad(single_log_likelihood)
        all_grad = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, X, y)
        fisher = jax.tree.map(lambda g: jnp.mean(g**2, axis=0), all_grad)
        epsilon = 1e-8
        fisher = jax.tree.map(lambda f: f / (jnp.max(f) + epsilon), fisher)
        return fisher
