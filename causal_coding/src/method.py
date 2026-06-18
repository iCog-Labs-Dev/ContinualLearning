import jax.numpy as jnp

from core.data import Task
from .training import train_step


class CausalCodingMethod:
    def __init__(
        self,
        lr_z,
        lr_w,
        lr_pi,
        num_inference_steps,
        gate_p,
        gate_kappa,
        ridge,
        lambda_s,
        batch_size,
        epochs,
    ):
        self.lr_z = lr_z
        self.lr_w = lr_w
        self.lr_pi = lr_pi
        self.num_inference_steps = num_inference_steps
        self.gate_p = gate_p
        self.gate_kappa = gate_kappa
        self.ridge = ridge
        self.lambda_s = lambda_s
        self.batch_size = batch_size
        self.epochs = epochs

    def train_task(self, model, params, state, task: Task, task_idx):
        num_classes = model.layer_sizes[-1]
        y_onehot = jnp.eye(num_classes)[task.train_y]
        num_batch = task.train_X.shape[0] // self.batch_size

        for ep in range(self.epochs):
            total_loss = 0.0
            for i in range(num_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                batch_X = task.train_X[start:end]
                batch_y = y_onehot[start:end]

                params, loss = train_step(
                    params,
                    batch_X,
                    batch_y,
                    self.num_inference_steps,
                    self.lr_z,
                    self.lr_w,
                    self.lr_pi,
                    self.gate_p,
                    self.gate_kappa,
                    self.ridge,
                    self.lambda_s,
                )
                total_loss += loss

            print(f"Epoch {ep + 1} ========= Loss: {total_loss / num_batch}")

        return params, state, total_loss / num_batch
