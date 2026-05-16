import os
import sys
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.data import load_mnist, split_into_tasks
from core.model import MLP
from core.runner import evaluate
from core.metrics import backward_transfer, forward_transfer, average_accuracy
from core.base import CausalState
from causal_coding.method import CausalMethod
from causal_coding.metrics import gate_jaccard, commutator_proxy

X, y, test_X, test_y = load_mnist()
class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)

model = MLP([784, 512, 512, 10])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

causal_method = CausalMethod(
    lr=0.01, batch_size=128, epochs=10, p=2.0, kappa=1e-8, lambda_clarity=0.1,
    use_protection=True, gate_quantile=0.90, support_frac=0.15,
    influence_mode="composite", use_head_protection=False,
)
params_c = params
state_c = CausalState(
    old_params=None,
    influence_scores=None,
    gate_vectors=None,
    all_gate_vectors=[],
    accumulated_support={},
    seen_classes=[],
)

class_il_causal = []
task_il_causal = []
causal_states = []

for task_idx, task in enumerate(tasks):
    print(f"--- Training Task {task_idx + 1} ---")
    params_c, state_c, _ = causal_method.train_task(
        model, params_c, state_c, task, task_idx
    )
    causal_states.append(state_c)

    class_row = []
    task_row = []
    for eval_task in tasks:
        class_row.append(evaluate(model, params_c, eval_task))
        task_row.append(evaluate(model, params_c, eval_task, eval_task.classes))

    class_il_causal.append(class_row)
    task_il_causal.append(task_row)

    for i, (acc_cil, acc_til) in enumerate(zip(class_row, task_row)):
        print(
            f"  Task {i + 1} -> Class-IL: {acc_cil * 100:.2f}% | Task-IL: {acc_til * 100:.2f}%"
        )

print("\n=== CausalMethod ===")
print(f"Avg Class-IL: {average_accuracy(class_il_causal) * 100:.2f}%")
print(f"Avg Task-IL:  {average_accuracy(task_il_causal)  * 100:.2f}%")
print(f"BWT Class-IL: {backward_transfer(class_il_causal) * 100:.2f}%")
print(f"BWT Task-IL:  {backward_transfer(task_il_causal)  * 100:.2f}%")
print(f"FWT Class-IL: {forward_transfer(class_il_causal)  * 100:.2f}%")

print("\n=== Gate Support Overlap ===")

for i in range(5):
    for j in range(i + 1, 5):
        g_i = causal_states[i].gate_vectors
        g_j = causal_states[j].gate_vectors

        jac, _ = gate_jaccard(g_i, g_j)
        defect, _ = commutator_proxy(g_i, g_j)

        print(f"T{i} vs T{j} | Jaccard: {jac:.3f} | Commutator proxy:{defect:.3f}")
