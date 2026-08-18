from __future__ import annotations

import jax
import jax.numpy as jnp

from discrimunative.src.training_discriminative import train_step_discriminative
from discrimunative.src.inference_discriminative import settle_states_discriminative
from discrimunative.src.energy_discriminative import compute_total_energy_discriminative
from core.data import Task

from functools import partial

@partial(jax.jit, static_argnames=("eta_x", "inference_steps"))
def compute_energy_batch(params, X, Y, eta_x, inference_steps):
    states, _ = settle_states_discriminative(
        params, X, Y, num_steps=inference_steps, eta_x=eta_x, mode="bottom_up"
    )
    return jnp.mean(compute_total_energy_discriminative(params, states))


class PCNMethodDiscriminative:
    def __init__(
        self,
        layer_sizes: list[int],
        eta_x: float,
        eta_w: float,
        inference_steps: int,
        batch_size: int,
        epochs: int,
        seed: int,
    ):
        self.layer_sizes = layer_sizes
        self.eta_x = eta_x
        self.eta_w = eta_w
        self.inference_steps = inference_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        
        self.task_il_training = False

    def train_task(self, model, params, state, task: Task, task_idx: int):
        protocol = "Task-IL" if self.task_il_training else "Class-IL"
        print(
            f"\n[PCNMethodDiscriminative] Task {task_idx + 1} | {protocol} | "
            f"inference_steps={self.inference_steps} | "
            f"eta_x={self.eta_x} | eta_w={self.eta_w}"
        )

        X_train = task.train_X * 2.0 - 1.0
        num_classes = self.layer_sizes[-1]
        Y_oh = jax.nn.one_hot(task.train_y, num_classes)

        n = X_train.shape[0]
        num_batches = (n + self.batch_size - 1) // self.batch_size
        
        base_key = jax.random.fold_in(jax.random.PRNGKey(self.seed), task_idx)

        energy_before = 0.0
        for b in range(num_batches):
            start = b * self.batch_size
            end = min(start + self.batch_size, n)
            energy_before += float(compute_energy_batch(
                params, X_train[start:end], Y_oh[start:end], self.eta_x, self.inference_steps
            ))
        energy_before /= num_batches
        print(f"  Task {task_idx + 1} energy before training: {energy_before:.4f}")

        total_energy = 0.0

        for ep in range(self.epochs):
            ep_key = jax.random.fold_in(base_key, ep)
            perm = jax.random.permutation(ep_key, n)
            X_ep = X_train[perm]
            Y_ep = Y_oh[perm]

            ep_energy = 0.0
            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n)
                X_b = X_ep[start:end]
                Y_b = Y_ep[start:end]

                params, metrics = train_step_discriminative(
                    params,
                    X_b,
                    Y_b,
                    eta_x=self.eta_x,
                    eta_w=self.eta_w,
                    inference_steps=self.inference_steps,
                    init_mode="bottom_up",
                )
                ep_energy += float(metrics["energy"])

            avg_ep_energy = ep_energy / num_batches
            total_energy += avg_ep_energy

            print(
                f"  Epoch {ep + 1:>3}/{self.epochs} | "
                f"avg settled energy: {avg_ep_energy:.4f}"
            )

        energy_after = 0.0
        for b in range(num_batches):
            start = b * self.batch_size
            end = min(start + self.batch_size, n)
            energy_after += float(compute_energy_batch(
                params, X_train[start:end], Y_oh[start:end], self.eta_x, self.inference_steps
            ))
        energy_after /= num_batches
        print(f"  Task {task_idx + 1} energy after training:  {energy_after:.4f}")

        avg_energy = total_energy / self.epochs
        print(f"  Task {task_idx + 1} done. Mean train energy over epochs: {avg_energy:.4f}")
        return params, None, avg_energy
