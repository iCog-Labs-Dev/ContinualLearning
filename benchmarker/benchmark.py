import copy

from .evaluator import Evaluator
from .results import BenchmarkResults
from .metrics import average_accuracy, backward_transfer, forward_transfer, forgetting
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
        class_il_baselines, task_il_baselines = self._evaluator.compute_baselines(
            self.model, params, self.tasks
        )

        class_il_matrix = self._run_protocol(
            params, state, protocol_name="Class-IL", task_il_training=False
        )
        task_il_matrix = self._run_protocol(
            params, state, protocol_name="Task-IL", task_il_training=True
        )
        self.method.task_il_training = False

        metrics = self._compute_metrics(
            class_il_matrix, task_il_matrix, class_il_baselines, task_il_baselines
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
        save_results(self.name, metrics, class_il_matrix, task_il_matrix, self.config)
        plot_all(self.name, class_il_matrix, task_il_matrix)

        return results

    def _run_protocol(self, params, state, protocol_name, task_il_training):
        self.method.task_il_training = task_il_training
        run_params = params
        run_state = copy.deepcopy(state)
        matrix = []
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

            matrix.append(row)

            for i, acc in enumerate(row):
                print(f"  Task {i + 1} -> {protocol_name}: {acc * 100:.2f}%")

        return matrix

    def _compute_metrics(
        self, class_il_matrix, task_il_matrix, class_il_baselines, task_il_baselines
    ) -> dict:
        return {
            "task_il": {
                "average_accuracy": float(average_accuracy(task_il_matrix)),
                "backward_transfer": float(backward_transfer(task_il_matrix)),
                "forward_transfer": float(
                    forward_transfer(task_il_matrix, task_il_baselines)
                ),
                "forgetting": float(forgetting(task_il_matrix)),
            },
            "class_il": {
                "average_accuracy": float(average_accuracy(class_il_matrix)),
                "backward_transfer": float(backward_transfer(class_il_matrix)),
                "forward_transfer": float(
                    forward_transfer(class_il_matrix, class_il_baselines)
                ),
                "forgetting": float(forgetting(class_il_matrix)),
            },
        }
