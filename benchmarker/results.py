from dataclasses import dataclass, field


@dataclass
class BenchmarkResults:
    method_name: str
    config: dict
    class_il_matrix: list
    task_il_matrix: list
    metrics: dict
    class_il_baselines: list = field(default_factory=list)
    task_il_baselines: list = field(default_factory=list)

    def summary(self):
        til = self.metrics["task_il"]
        cil = self.metrics["class_il"]
        print(f"\n=== {self.method_name} ===")
        print(
            f"Task-IL  | ACC: {til['average_accuracy']:.4f} | NLL: {til['average_nll']:.4f}"
            f" | BWT: {til['backward_transfer']:.4f} | FWT: {til['forward_transfer']:.4f}"
            f" | Forgetting: {til['forgetting']:.4f}"
        )
        print(
            f"Class-IL | ACC: {cil['average_accuracy']:.4f} | NLL: {cil['average_nll']:.4f}"
            f" | BWT: {cil['backward_transfer']:.4f} | FWT: {cil['forward_transfer']:.4f}"
            f" | Forgetting: {cil['forgetting']:.4f}"
        )
