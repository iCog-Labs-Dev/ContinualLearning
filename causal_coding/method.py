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

    for layer_key in params:
        gate_w = gates[layer_key]["w"]
        gate_collapsed = jnp.mean(gate_w, axis=0)
        gate_matrix = gate_collapsed[:, None]

        gated_grads[layer_key] = {
            "w": grads[layer_key]["w"] * gate_matrix,
            "b": grads[layer_key]["b"] * gates[layer_key]["b"],
        }

    new_params = jax.tree.map(lambda param, g: param - lr * g, params, gated_grads)
    w_arrays = [gates[k]["w"] for k in gates]

    all_gates = jnp.concatenate([jnp.ravel(w) for w in w_arrays])

    sparsity = jnp.mean(all_gates < 0.05)

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


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from core.model import MLP

    key = jax.random.PRNGKey(0)
    model = MLP([4, 8, 8, 3])
    params = model.init_params(key)

    key, subkey = jax.random.split(key)
    batch_X = jax.random.normal(subkey, shape=(16, 4))
    batch_y = jax.random.randint(key, shape=(16,), minval=0, maxval=3)

    new_params, loss, sparsity = _gated_train_step(
        params, batch_X, batch_y, lr=0.01, p=1.0, kappa=1e-8, model=model
    )

    print("loss:           ", loss)
    print("sparsity:       ", sparsity)
    print(
        "params changed: ",
        not jnp.allclose(params["layer_1"]["w"], new_params["layer_1"]["w"]),
    )
    print("no NaNs:        ", not jnp.any(jnp.isnan(new_params["layer_1"]["w"])))

    task = Task(
        train_X=jax.random.normal(key, shape=(64, 4)),
        train_y=jax.random.randint(key, shape=(64,), minval=0, maxval=3),
        test_X=jax.random.normal(key, shape=(20, 4)),
        test_y=jax.random.randint(key, shape=(20,), minval=0, maxval=3),
        classes=[0, 1, 2],
    )

    method = CausalMethod(lr=0.01, batch_size=16, epochs=3, p=1.0, kappa=1e-8)
    params, causal_state, final_loss = method.train_task(model, params, None, task, 0)

    print("CausalState type:      ", type(causal_state).__name__)
    print("old_params has layer_1:", "layer_1" in causal_state.old_params)
    print("influence_scores keys: ", list(causal_state.influence_scores.keys()))
