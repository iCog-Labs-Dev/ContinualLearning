from .metrics import average_accuracy, backward_transfer, forward_transfer, forgetting
from .logger import save_results
from .plotter import plot_all


def benchmark(method_name, class_il_matrix, task_il_matrix, baselines):

    metrics_dict = {
        "task_il": {
            "average_accuracy": float(average_accuracy(task_il_matrix)),
            "backward_transfer": float(backward_transfer(task_il_matrix)),
            "forward_transfer": float(forward_transfer(task_il_matrix, baselines)),
            "forgetting": float(forgetting(task_il_matrix)),
        },
        "class_il": {
            "average_accuracy": float(average_accuracy(class_il_matrix)),
            "backward_transfer": float(backward_transfer(class_il_matrix)),
            "forward_transfer": float(forward_transfer(class_il_matrix, baselines)),
            "forgetting": float(forgetting(class_il_matrix)),
        },
    }

    print(
        f"Task-IL| ACC: {metrics_dict['task_il']['average_accuracy']} | BWT: {metrics_dict['task_il']['backward_transfer']} | FWT: {metrics_dict['task_il']['forward_transfer']} | Forgetting: {metrics_dict['task_il']['forgetting']}"
    )
    print(
        f"\nClass-IL| ACC: {metrics_dict['class_il']['average_accuracy']} | BWT: {metrics_dict['class_il']['backward_transfer']} | FWT: {metrics_dict['class_il']['forward_transfer']} | Forgetting: {metrics_dict['class_il']['forgetting']}"
    )

    save_results(method_name, metrics_dict, class_il_matrix, task_il_matrix)
    plot_all(method_name, class_il_matrix, task_il_matrix)
