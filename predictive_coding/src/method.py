"""
PCNMethod — benchmark adapter for the Generative Predictive Coding Network.

This class integrates a purely generative Predictive Coding Network (PCN)
with the continual-learning benchmark framework. It exposes the standard
`train_task(...)` interface while preserving the PCN training procedure:
iterative state inference followed by local prediction-error-driven weight
updates.

The benchmark data is automatically rescaled from [0,1] to [-1,1] to match
the tanh-based PCN formulation, and integer labels are converted to one-hot
targets. No replay buffer, regularisation, or task-specific memory is used;
the model is trained sequentially on tasks using only predictive-coding
dynamics.


"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from predictive_coding.src.training import train_step
from predictive_coding.src.inference import settle_states
from predictive_coding.src.energy import compute_total_energy
from core.data import Task

from functools import partial

@partial(jax.jit, static_argnames=("eta_x", "inference_steps"))
def compute_energy_batch(params, X, Y, eta_x, inference_steps):
    states, _ = settle_states(
        params, X, Y, num_steps=inference_steps, eta_x=eta_x, mode="bottom_up"
    )
    return jnp.mean(compute_total_energy(params, states))


class PCNMethod:
    """Benchmark adapter for a purely generative Predictive Coding Network.

    Parameters
    ----------
    layer_sizes:
        Architecture
    eta_x:
        Inference (state-settling) learning rate.
    eta_w:
        Weight learning rate.
    inference_steps:
        Number of settling steps per training batch. Passed through from
        ``SuiteConfig.inference_steps`` so it can be controlled via CLI
        without touching code.
    batch_size:
        Mini-batch size.
    epochs:
        Training epochs per task.
    seed:
        Base PRNG seed. Per-task and per-epoch keys are derived from this.
    """

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

    # Benchmark interface
  

    def train_task(self, model, params, state, task: Task, task_idx: int):
        """Train on one task sequentially """


        protocol = "Task-IL" if self.task_il_training else "Class-IL"
        print(
            f"\n[PCNMethod] Task {task_idx + 1} | {protocol} | "
            f"inference_steps={self.inference_steps} | "
            f"eta_x={self.eta_x} | eta_w={self.eta_w}"
        )

        # rescale: benchmark data is [0,1]; PCN was validated on [-1,1]
        X_train = task.train_X * 2.0 - 1.0          # shape: (N, 784)

        # one-hot encode integer labels 
        num_classes = self.layer_sizes[-1]           # typically 10
        Y_oh = jax.nn.one_hot(task.train_y, num_classes)  # (N, 10)

        n = X_train.shape[0]
        num_batches = (n + self.batch_size - 1) // self.batch_size
        

        # Derive a per-task base key so tasks don't share shuffle state.
        base_key = jax.random.fold_in(
            jax.random.PRNGKey(self.seed), task_idx
        )

        # Energy before training this task
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

            # Shuffle
            perm = jax.random.permutation(ep_key, n)
            X_ep = X_train[perm]
            Y_ep = Y_oh[perm]

            ep_energy = 0.0
            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n)
                X_b = X_ep[start:end]
                Y_b = Y_ep[start:end]

                params, metrics = train_step(
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

        # Energy after training this task
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
