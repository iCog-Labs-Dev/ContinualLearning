import os
import sys
import jax
import jax.numpy as jnp
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.model import MLP
from core.metrics import cross_entropy
from causal_coding.gating import estimate_influence, compute_gates
from core.data import Task
from core.base import CausalState


def _loss_fn(params, X, y, model: MLP):
    logits = model.forward(params, X)
    return cross_entropy(logits, y)


@partial(jax.jit, static_argnums=(6,))
def _gated_train_step(params, X, y, lr, p, kappa, model: MLP):
    loss, grads = jax.value_and_grad(_loss_fn)(params, X, y, model)
    pre_acts, _, _ = model.forward_with_states(params, X)
    # compute the influence and gates
    influence = estimate_influence(params, pre_acts, batch_size=X.shape[0])
    gates = compute_gates(influence, p, kappa)
    gated_grads = {}
    final_gates = []

    for layer_key in params:
        gate_w = gates[layer_key]["w"]
        gate_collapsed = jnp.mean(gate_w, axis=0)
        gate_normalized = gate_collapsed / (jnp.mean(gate_collapsed) + 1e-8)
        gate_final = jnp.minimum(gate_normalized, 1.0)
        gate_matrix = gate_final[:, None]

        gated_grads[layer_key] = {
            "w": grads[layer_key]["w"] * gate_matrix,
            "b": grads[layer_key]["b"] * gates[layer_key]["b"],
        }

        final_gates.append(jnp.ravel(gate_final))

    new_params = jax.tree.map(lambda param, g: param - lr * g, params, gated_grads)

    all_gates = jnp.concatenate(final_gates)

    sparsity = jnp.mean(all_gates < 0.5)

    return new_params, loss, sparsity


class CausalMethod:
    def __init__(self, lr, batch_size, epochs, p, kappa):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.p = p
        self.kappa = kappa

    def train_task(self, model: MLP, params, state, task: Task, task_idx):
        num_batches = task.train_X.shape[0] // self.batch_size

        for epoch in range(self.epochs):
            total_loss = 0
            total_sparsity = 0

            for i in range(num_batches):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                batch_X = task.train_X[start:end]
                batch_y = task.train_y[start:end]

                params, loss, sparsity = _gated_train_step(
                    params, batch_X, batch_y, self.lr, self.p, self.kappa, model
                )
                total_loss += loss
                total_sparsity += sparsity

            print(
                f"Epoch: {epoch + 1} Loss: {total_loss/num_batches:.4f} | Sparsity: {total_sparsity/num_batches:.4f}"
            )

        pre_acts, _, _ = model.forward_with_states(params, batch_X)
        final_influence = estimate_influence(params, pre_acts, self.batch_size)

        causal_state = CausalState(old_params=params, influence_scores=final_influence)

        return params, causal_state, total_loss / num_batches
