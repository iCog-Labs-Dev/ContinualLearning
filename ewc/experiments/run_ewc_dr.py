import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import MLP
from src.data import Task, load_mnist, split_into_tasks
from src.utils import average_accuracy, plot_accuracy_matrix, backward_transfer
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

class_il_matrix = []
task_il_matrix = []

state = {"anchors": []}

for task_idx in range(len(class_pairs)):
    print(f"\n--- Training Task {task_idx + 1} ---")
    params, state, loss = method.train_task(
        model, params, state, tasks[task_idx], task_idx
    )

    class_il_accuracies = []
    task_il_accuracies = []

    for eval_idx in range(len(class_pairs)):
        acc_cil = method.evaluate(model, params, tasks[eval_idx], allowed_classes=None)
        class_il_accuracies.append(acc_cil)

        acc_til = method.evaluate(
            model, params, tasks[eval_idx], allowed_classes=tasks[eval_idx].classes
        )
        task_il_accuracies.append(acc_til)

    class_il_matrix.append(class_il_accuracies)
    task_il_matrix.append(task_il_accuracies)

    for i, (acc_cil, acc_til) in enumerate(
        zip(class_il_accuracies, task_il_accuracies)
    ):
        print(
            f"Eval on Task {i + 1} -> Class-IL: {acc_cil * 100:.2f}% | Task-IL: {acc_til * 100:.2f}%"
        )

print(f"\nAverage Class-IL Accuracy: {average_accuracy(class_il_matrix) * 100:.2f}%")
print(f"Average Task-IL Accuracy: {average_accuracy(task_il_matrix) * 100:.2f}%")
print(
    f"Backward Transfer of Class-IL: {backward_transfer(class_il_matrix)} and Task-IL {backward_transfer(task_il_matrix)}"
)
