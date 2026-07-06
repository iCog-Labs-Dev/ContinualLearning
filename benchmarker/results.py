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
            "           NLL-space | "
            f"BWT: {til['backward_transfer_nll']:.4f} | BWT(best): {til['bwt_nll']:.4f}"
            f" | FWT: {til['forward_transfer_nll']:.4f}"
            f" | Forgetting: {til['forgetting_nll']:.4f}"
        )
        print("           per-task NLL: " + ", ".join(f"{v:.4f}" for v in til["per_task_nll"]))
        print(
            f"           BCE-NLL avg: {til['average_bce_nll']:.4f} | per-task: "
            + ", ".join(f"{v:.4f}" for v in til["per_task_bce_nll"])
        )

        print(
            f"Class-IL | ACC: {cil['average_accuracy']:.4f} | NLL: {cil['average_nll']:.4f}"
            f" | BWT: {cil['backward_transfer']:.4f} | FWT: {cil['forward_transfer']:.4f}"
            f" | Forgetting: {cil['forgetting']:.4f}"
        )
        print(
            "           NLL-space | "
            f"BWT: {cil['backward_transfer_nll']:.4f} | BWT(best): {cil['bwt_nll']:.4f}"
            f" | FWT: {cil['forward_transfer_nll']:.4f}"
            f" | Forgetting: {cil['forgetting_nll']:.4f}"
        )
        print("           per-task NLL: " + ", ".join(f"{v:.4f}" for v in cil["per_task_nll"]))
