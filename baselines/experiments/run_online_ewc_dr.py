import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import Task, load_mnist, split_into_tasks
from core.metrics import average_accuracy, plot_accuracy_matrix, backward_transfer
from core.base import EWCState
from src.ewc_dr import EWCDRMethod

X, y, test_X, test_y = load_mnist()
class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)

model = MLP([784, 512, 512, 10])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

method = EWCDRMethod(
    lr=0.001,
    lr_task1=0.01,
    batch_size=128,
    epochs=50,
    lam=1000,
    num_samples=200,
    decay=0.9,
)

accuracy_matrix = []

state = EWCState(
    old_params=params,
    cumulative_fisher=jax.tree.map(lambda p: jnp.zeros_like(p), params),
)

for task_idx in range(len(class_pairs)):
    print(f"Training Task {task_idx + 1}")
    params, state, loss = method.train_task(
        model, params, state, tasks[task_idx], task_idx
    )

    task_accuracies = []
    for eval_idx in range(len(class_pairs)):
        acc = method.evaluate(model, params, tasks[eval_idx])
        task_accuracies.append(acc)

    accuracy_matrix.append(task_accuracies)

    for i, acc in enumerate(task_accuracies):
        print(f"Task {i + 1}: accuracy: {acc * 100}%")

print(f"Average Accuracy: {average_accuracy(accuracy_matrix) * 100}%")
print(f"Backward Transfer: {backward_transfer(accuracy_matrix) * 100}%")
plot_accuracy_matrix(
    accuracy_matrix, "online EWC Done Right", "plots/online_ewc_dr.png"
)
