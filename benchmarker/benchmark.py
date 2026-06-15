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
        self.config = config if config is not None else vars(method)
        self._evaluator = Evaluator()

    def run(self, params, state) -> BenchmarkResults:
        class_il_baselines, task_il_baselines = self._evaluator.compute_baselines(
            self.model, params, self.tasks
        )

        class_il_matrix = []
        task_il_matrix = []
        class_il_nll_matrix = []
        task_il_nll_matrix = []

        for task_idx, task in enumerate(self.tasks):
            print(f"\n--- Training Task {task_idx + 1} ---")
            params, state, _ = self.method.train_task(
                self.model, params, state, task, task_idx
            )

            class_il_row, task_il_row, class_il_nll_row, task_il_nll_row = (
                self._evaluator.evaluate_all(self.model, params, self.tasks)
            )
            class_il_matrix.append(class_il_row)
            task_il_matrix.append(task_il_row)
            class_il_nll_matrix.append(class_il_nll_row)
            task_il_nll_matrix.append(task_il_nll_row)

            for i, (cil, til, cil_nll, til_nll) in enumerate(
                zip(class_il_row, task_il_row, class_il_nll_row, task_il_nll_row)
            ):
                print(
                    f"  Task {i + 1} -> "
                    f"Class-IL: {cil * 100:.2f}% (NLL {cil_nll:.3f}) | "
                    f"Task-IL: {til * 100:.2f}% (NLL {til_nll:.3f})"
                )

        metrics = self._compute_metrics(
            class_il_matrix,
            task_il_matrix,
            class_il_nll_matrix,
            task_il_nll_matrix,
            class_il_baselines,
            task_il_baselines,
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

    def _compute_metrics(
        self,
        class_il_matrix,
        task_il_matrix,
        class_il_nll_matrix,
        task_il_nll_matrix,
        class_il_baselines,
        task_il_baselines,
    ) -> dict:
        return {
            "task_il": {
                "average_accuracy": float(average_accuracy(task_il_matrix)),
                "average_nll": float(average_accuracy(task_il_nll_matrix)),
                "backward_transfer": float(backward_transfer(task_il_matrix)),
                "forward_transfer": float(
                    forward_transfer(task_il_matrix, task_il_baselines)
                ),
                "forgetting": float(forgetting(task_il_matrix)),
            },
            "class_il": {
                "average_accuracy": float(average_accuracy(class_il_matrix)),
                "average_nll": float(average_accuracy(class_il_nll_matrix)),
                "backward_transfer": float(backward_transfer(class_il_matrix)),
                "forward_transfer": float(
                    forward_transfer(class_il_matrix, class_il_baselines)
                ),
                "forgetting": float(forgetting(class_il_matrix)),
            },
        }
