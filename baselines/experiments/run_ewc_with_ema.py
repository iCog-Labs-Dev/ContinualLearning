import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.base import EWCState
from benchmarker.evaluator import Evaluator
from benchmarker.metrics import average_accuracy, backward_transfer, forward_transfer, forgetting
from benchmarker.logger import save_results
from benchmarker.plotter import plot_all
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
    anchor_alpha=0.5,
)
_evaluator = Evaluator()

class_il_baselines, task_il_baselines = _evaluator.compute_baselines(model, params, tasks)


def _initial_state():
    return EWCState(
        old_params=params,
        cumulative_fisher=jax.tree.map(lambda p: jnp.zeros_like(p), params),
    )


def _run_protocol(protocol_name, task_il_training):
    method.task_il_training = task_il_training
    run_params = params
    run_state = _initial_state()
    ema_params = params
    matrix = []
    train_loss = "BCE" if task_il_training else "CE"
    eval_mode = "sigmoid" if task_il_training else "softmax"

    print(
        f"\n=== {protocol_name} run "
        f"({train_loss} training / {eval_mode} evaluation) ==="
    )

    for task_idx, task in enumerate(tasks):
        print(f"\n--- Training Task {task_idx + 1} ---")
        run_params, run_state, _ = method.train_task(
            model, run_params, run_state, task, task_idx
        )

        ema_params = jax.tree.map(
            lambda ema, new: method.anchor_alpha * ema + (1 - method.anchor_alpha) * new,
            ema_params,
            run_params,
        )

        if task_il_training:
            row = [_evaluator.evaluate(model, ema_params, t, t.classes) for t in tasks]
        else:
            row = [_evaluator.evaluate(model, ema_params, t) for t in tasks]

        matrix.append(row)

        for i, acc in enumerate(row):
            print(f"  Task {i + 1} -> {protocol_name}: {acc * 100:.2f}%")

    return matrix


class_il_matrix = _run_protocol("Class-IL", task_il_training=False)
task_il_matrix = _run_protocol("Task-IL", task_il_training=True)
method.task_il_training = False
config = dict(vars(method))
config.pop("task_il_training", None)
config["protocols"] = {
    "class_il": "CE training / softmax evaluation",
    "task_il": "BCE training / sigmoid evaluation",
}

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

save_results("ewc_with_ema", metrics, class_il_matrix, task_il_matrix, config=config)
plot_all("ewc_with_ema", class_il_matrix, task_il_matrix)
