import jax.numpy as jnp

from core.data import Task
from core.model import MLP


def evaluate(model: MLP, params, task: Task, allowed_classes=None):
    logits = model.forward(params, task.test_X)

    if allowed_classes is not None:
        mask = jnp.full((logits.shape[1],), -jnp.inf)
        mask = mask.at[jnp.array(allowed_classes)].set(0.0)
        logits = logits + mask

    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == task.test_y)


def run_experiment(method, model: MLP, params, state, tasks):
    class_il_matrix = []
    task_il_matrix = []

    for task_idx, task in enumerate(tasks):
        print(f"\n--- Training Task {task_idx + 1} ---")
        params, state, _ = method.train_task(model, params, state, task, task_idx)

        class_il_row = []
        task_il_row = []

        for eval_task in tasks:
            acc_cil = evaluate(model, params, eval_task)
            acc_til = evaluate(model, params, eval_task, eval_task.classes)
            class_il_row.append(acc_cil)
            task_il_row.append(acc_til)

        class_il_matrix.append(class_il_row)
        task_il_matrix.append(task_il_row)

        for i, (acc_cil, acc_til) in enumerate(zip(class_il_row, task_il_row)):
            print(
                f"  Task {i + 1} -> Class-IL: {acc_cil * 100:.2f}% | Task-IL: {acc_til * 100:.2f}%"
            )

    return params, state, class_il_matrix, task_il_matrix
