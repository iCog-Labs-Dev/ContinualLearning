import os
import json
import csv


def _save_json(output_dir, method_name, metrics_dict):
    data = {
        "method": method_name,
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


def save_results(method_name, metrics_dict, class_il_matrix, task_il_matrix):
    output_dir = "results/" + method_name
    os.makedirs(output_dir, exist_ok=True)

    _save_json(output_dir, method_name, metrics_dict)
    _save_matrix_csv(output_dir, class_il_matrix, "class_il_matrix.csv")
    _save_matrix_csv(output_dir, task_il_matrix, "task_il_matrix.csv")

    print(f"Logs saved in {output_dir}")
