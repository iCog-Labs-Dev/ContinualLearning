import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.metrics import average_accuracy, plot_accuracy_matrix
from core.base import EWCState
from core.runner import run_experiment
from src.ewc_dr import EWCDRMethod

X, y, test_X, test_y = load_mnist()
class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)

model = MLP([784, 512, 512, 10])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

method = EWCDRMethod(
    lr=0.001, lr_task1=0.01, batch_size=128, epochs=25, lam=100, num_samples=200
)
state = EWCState(
    old_params=params,
    cumulative_fisher=jax.tree.map(lambda p: jnp.zeros_like(p), params),
)

params, _, class_il_matrix, task_il_matrix = run_experiment(method, model, params, state, tasks)

print(f"\nAverage Class-IL Accuracy: {average_accuracy(class_il_matrix) * 100:.2f}%")
print(f"Average Task-IL Accuracy: {average_accuracy(task_il_matrix) * 100:.2f}%")

plot_accuracy_matrix(class_il_matrix, "EWC Done Right (Class-IL)", "plots/ewc_dr_class_il.png")
plot_accuracy_matrix(task_il_matrix, "EWC Done Right (Task-IL)", "plots/ewc_dr_task_il.png")
