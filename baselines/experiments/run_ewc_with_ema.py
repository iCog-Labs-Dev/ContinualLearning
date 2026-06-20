import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.base import EWCState
from core.config import get_config
from benchmarker.evaluator import Evaluator
from benchmarker.metrics import average_accuracy, backward_transfer, forward_transfer, forgetting
from benchmarker.logger import save_results
from benchmarker.plotter import plot_all
from src.ewc_dr import EWCDRMethod

X, y, test_X, test_y = load_mnist()
# Load hyperparameters from YAML or fallback to defaults
config = get_config(
    default_method_kwargs=dict(lr=0.001, lr_task1=0.01, batch_size=128, epochs=50, lam=1000, num_samples=200, decay=0.9, anchor_alpha=0.5)
)
tasks = split_into_tasks(X, y, test_X, test_y, config.task.class_pairs)

# Initialize model using config dimensions
model = MLP([config.model.input_dim] + config.model.hidden_dims + [config.model.output_dim])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

# Inject kwargs directly into the method
method = EWCDRMethod(**config.method_kwargs)
state = EWCState(
    old_params=params,
    cumulative_fisher=jax.tree.map(lambda p: jnp.zeros_like(p), params),
)

_evaluator = Evaluator()

class_il_baselines, task_il_baselines = _evaluator.compute_baselines(model, params, tasks)

ema_params = params
class_il_matrix = []
task_il_matrix = []

for task_idx, task in enumerate(tasks):
    print(f"\n--- Training Task {task_idx + 1} ---")
    params, state, _ = method.train_task(model, params, state, task, task_idx)

    ema_params = jax.tree.map(
        lambda ema, new: method.anchor_alpha * ema + (1 - method.anchor_alpha) * new,
        ema_params,
        params,
    )

    class_il_row = [_evaluator.evaluate(model, ema_params, t) for t in tasks]
    task_il_row = [_evaluator.evaluate(model, ema_params, t, t.classes) for t in tasks]

    class_il_matrix.append(class_il_row)
    task_il_matrix.append(task_il_row)

    for i, (acc_cil, acc_til) in enumerate(zip(class_il_row, task_il_row)):
        print(
            f"  Task {i + 1} -> Class-IL: {acc_cil * 100:.2f}% | Task-IL: {acc_til * 100:.2f}%"
        )

metrics = {
    "task_il": {
        "average_accuracy": float(average_accuracy(task_il_matrix)),
        "backward_transfer": float(backward_transfer(task_il_matrix)),
        "forward_transfer": float(forward_transfer(task_il_matrix, task_il_baselines)),
        "forgetting": float(forgetting(task_il_matrix)),
    },
    "class_il": {
        "average_accuracy": float(average_accuracy(class_il_matrix)),
        "backward_transfer": float(backward_transfer(class_il_matrix)),
        "forward_transfer": float(forward_transfer(class_il_matrix, class_il_baselines)),
        "forgetting": float(forgetting(class_il_matrix)),
    },
}

print(f"\n=== ewc_with_ema ===")
til, cil = metrics["task_il"], metrics["class_il"]
print(f"Task-IL  | ACC: {til['average_accuracy']:.4f} | BWT: {til['backward_transfer']:.4f} | FWT: {til['forward_transfer']:.4f} | Forgetting: {til['forgetting']:.4f}")
print(f"Class-IL | ACC: {cil['average_accuracy']:.4f} | BWT: {cil['backward_transfer']:.4f} | FWT: {cil['forward_transfer']:.4f} | Forgetting: {cil['forgetting']:.4f}")

save_results("ewc_with_ema", metrics, class_il_matrix, task_il_matrix, config=vars(method))
plot_all("ewc_with_ema", class_il_matrix, task_il_matrix)
