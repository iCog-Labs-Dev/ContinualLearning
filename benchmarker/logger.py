import os
import json
import csv


def _save_json(output_dir, method_name, metrics_dict, config=None):
    data = {
        "method": method_name,
        "config": config or {},
        "protocol": {
            "task_il": metrics_dict["task_il"],
            "class_il": metrics_dict["class_il"],
        },
    }

    path = output_dir + "/metrics.json"

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _save_matrix_csv(output_dir, matrix, filename):
    T = len(matrix)

    header = ["after_task"] + [f"task_{i}" for i in range(T)]
    path = output_dir + "/" + filename

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(T):
            row_label = f"after_task_{t}"
            values = [float(v) for v in matrix[t]]
            writer.writerow([row_label] + values)


def save_results(
    method_name,
    metrics_dict,
    class_il_matrix,
    task_il_matrix,
    config=None,
    class_il_nll_matrix=None,
    task_il_nll_matrix=None,
    task_il_bce_matrix=None,
):
    output_dir = "results/" + method_name
    os.makedirs(output_dir, exist_ok=True)

    _save_json(output_dir, method_name, metrics_dict, config)
    _save_matrix_csv(output_dir, class_il_matrix, "class_il_matrix.csv")
    _save_matrix_csv(output_dir, task_il_matrix, "task_il_matrix.csv")

    # Persist full T×T NLL / BCE-NLL trajectories; aggregate metrics keep only
    # the final row.
    if class_il_nll_matrix is not None:
        _save_matrix_csv(output_dir, class_il_nll_matrix, "class_il_nll_matrix.csv")
    if task_il_nll_matrix is not None:
        _save_matrix_csv(output_dir, task_il_nll_matrix, "task_il_nll_matrix.csv")
    if task_il_bce_matrix is not None:
        _save_matrix_csv(output_dir, task_il_bce_matrix, "task_il_bce_matrix.csv")

    print(f"Logs saved in {output_dir}")
