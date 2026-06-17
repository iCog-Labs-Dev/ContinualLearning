import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.metrics import average_accuracy, backward_transfer, plot_accuracy_matrix
from core.base import EWCState
from core.config import get_config
from core.runner import run_experiment
from src.ewc_with_ema import EWCEMAMethod

X, y, test_X, test_y = load_mnist()
# Load hyperparameters from YAML or fallback to defaults
config = get_config(
    default_method_kwargs=dict(lr=0.001, batch_size=128, epochs=50)
)
tasks = split_into_tasks(X, y, test_X, test_y, config.task.class_pairs)

# Initialize model using config dimensions
model = MLP([config.model.input_dim] + config.model.hidden_dims + [config.model.output_dim])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

# Inject kwargs directly into the method
method = EWCEMAMethod(**config.method_kwargs)
state = EWCState(
    old_params=params,
    cumulative_fisher=jax.tree.map(lambda p: jnp.zeros_like(p), params),
)

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

    class_il_row = []
    task_il_row = []

    for eval_task in tasks:
        acc_cil = evaluate(model, ema_params, eval_task)
        acc_til = evaluate(model, ema_params, eval_task, eval_task.classes)
        class_il_row.append(acc_cil)
        task_il_row.append(acc_til)

    class_il_matrix.append(class_il_row)
    task_il_matrix.append(task_il_row)

    for i, (acc_cil, acc_til) in enumerate(zip(class_il_row, task_il_row)):
        print(
            f"  Task {i + 1} -> Class-IL: {acc_cil * 100:.2f}% | Task-IL: {acc_til * 100:.2f}%"
        )

print(f"\nAverage Class-IL Accuracy: {average_accuracy(class_il_matrix) * 100:.2f}%")
print(f"Average Task-IL Accuracy: {average_accuracy(task_il_matrix) * 100:.2f}%")
print(f"Backward Transfer (Class-IL): {backward_transfer(class_il_matrix) * 100:.2f}%")
