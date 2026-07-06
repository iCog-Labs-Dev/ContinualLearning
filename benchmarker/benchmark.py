import copy

from .evaluator import Evaluator
from .results import BenchmarkResults
from .metrics import (
    average_accuracy,
    backward_transfer,
    forward_transfer,
    forgetting,
    backward_transfer_nll,
    bwt_nll,
    forgetting_nll,
    forward_transfer_nll,
)
from .logger import save_results
from .plotter import plot_all


class CLBenchmark:

    def __init__(self, method, model, tasks, name, config=None):
        self.method = method
        self.model = model
        self.tasks = tasks
        self.name = name
        self.config = dict(config) if config is not None else dict(vars(method))
        self.config.pop("task_il_training", None)
        self.config["protocols"] = {
            "class_il": "CE training / softmax evaluation",
            "task_il": "BCE training / sigmoid evaluation",
        }
        self._evaluator = Evaluator()

    def run(self, params, state) -> BenchmarkResults:
        (
            class_il_baselines,
            task_il_baselines,
            class_il_nll_baselines,
            task_il_nll_baselines,
        ) = self._evaluator.compute_baselines(self.model, params, self.tasks)

        class_il_matrix, class_il_nll_matrix, _ = self._run_protocol(
            params, state, protocol_name="Class-IL", task_il_training=False
        )
        task_il_matrix, task_il_nll_matrix, task_il_bce_matrix = self._run_protocol(
            params, state, protocol_name="Task-IL", task_il_training=True
        )
        self.method.task_il_training = False

        metrics = self._compute_metrics(
            class_il_matrix,
            task_il_matrix,
            class_il_nll_matrix,
            task_il_nll_matrix,
            task_il_bce_matrix,
            class_il_baselines,
            task_il_baselines,
            class_il_nll_baselines,
            task_il_nll_baselines,
        )

        results = BenchmarkResults(
            method_name=self.name,
            config=self.config,
            class_il_matrix=class_il_matrix,
            task_il_matrix=task_il_matrix,
            metrics=metrics,
            class_il_baselines=class_il_baselines,
            task_il_baselines=task_il_baselines,
        )

        results.summary()
        save_results(
            self.name,
            metrics,
            class_il_matrix,
            task_il_matrix,
            self.config,
            class_il_nll_matrix=class_il_nll_matrix,
            task_il_nll_matrix=task_il_nll_matrix,
            task_il_bce_matrix=task_il_bce_matrix,
        )
        plot_all(self.name, class_il_matrix, task_il_matrix)

        return results

    def _run_protocol(self, params, state, protocol_name, task_il_training):
        self.method.task_il_training = task_il_training
        run_params = params
        run_state = copy.deepcopy(state)
        acc_matrix = []
        nll_matrix = []
        # Secondary Bernoulli BCE-NLL matrix is Task-IL only.
        bce_matrix = [] if task_il_training else None
        train_loss = "BCE" if task_il_training else "CE"
        eval_mode = "sigmoid" if task_il_training else "softmax"

        print(
            f"\n=== {protocol_name} run "
            f"({train_loss} training / {eval_mode} evaluation) ==="
        )

        for task_idx, task in enumerate(self.tasks):
            print(f"\n--- Training Task {task_idx + 1} ---")
            run_params, run_state, _ = self.method.train_task(
                self.model, run_params, run_state, task, task_idx
            )

            if task_il_training:
                row = [
                    self._evaluator.evaluate(self.model, run_params, t, t.classes)
                    for t in self.tasks
                ]
            else:
                row = [
                    self._evaluator.evaluate(self.model, run_params, t)
                    for t in self.tasks
                ]

            acc_row = [acc for acc, _, _ in row]
            nll_row = [nll for _, nll, _ in row]
            acc_matrix.append(acc_row)
            nll_matrix.append(nll_row)
            if task_il_training:
                bce_matrix.append([bce for _, _, bce in row])

            for i, (acc, nll, _bce) in enumerate(row):
                print(
                    f"  Task {i + 1} -> {protocol_name}: "
                    f"{acc * 100:.2f}% (NLL {nll:.3f})"
                )

        return acc_matrix, nll_matrix, bce_matrix

    def _compute_metrics(
        self,
        class_il_matrix,
        task_il_matrix,
        class_il_nll_matrix,
        task_il_nll_matrix,
        task_il_bce_matrix,
        class_il_baselines,
        task_il_baselines,
        class_il_nll_baselines,
        task_il_nll_baselines,
    ) -> dict:
        task_il = {
            "average_accuracy": float(average_accuracy(task_il_matrix)),
            "backward_transfer": float(backward_transfer(task_il_matrix)),
            "forward_transfer": float(
                forward_transfer(task_il_matrix, task_il_baselines)
            ),
            "forgetting": float(forgetting(task_il_matrix)),
            "average_nll": float(average_accuracy(task_il_nll_matrix)),
            "per_task_nll": [float(v) for v in task_il_nll_matrix[-1]],
            "backward_transfer_nll": float(backward_transfer_nll(task_il_nll_matrix)),
            "bwt_nll": float(bwt_nll(task_il_nll_matrix)),
            "forgetting_nll": float(forgetting_nll(task_il_nll_matrix)),
            "forward_transfer_nll": float(
                forward_transfer_nll(task_il_nll_matrix, task_il_nll_baselines)
            ),
            "average_bce_nll": float(average_accuracy(task_il_bce_matrix)),
            "per_task_bce_nll": [float(v) for v in task_il_bce_matrix[-1]],
        }
        class_il = {
            "average_accuracy": float(average_accuracy(class_il_matrix)),
            "backward_transfer": float(backward_transfer(class_il_matrix)),
            "forward_transfer": float(
                forward_transfer(class_il_matrix, class_il_baselines)
            ),
            "forgetting": float(forgetting(class_il_matrix)),
            "average_nll": float(average_accuracy(class_il_nll_matrix)),
            "per_task_nll": [float(v) for v in class_il_nll_matrix[-1]],
            "backward_transfer_nll": float(backward_transfer_nll(class_il_nll_matrix)),
            "bwt_nll": float(bwt_nll(class_il_nll_matrix)),
            "forgetting_nll": float(forgetting_nll(class_il_nll_matrix)),
            "forward_transfer_nll": float(
                forward_transfer_nll(class_il_nll_matrix, class_il_nll_baselines)
            ),
        }
        return {"task_il": task_il, "class_il": class_il}
