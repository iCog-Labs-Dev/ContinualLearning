import os
import matplotlib.pyplot as plt
import seaborn
import numpy as np


def plot_accuracy_matrix(matrix, title, output_dir):
    data = np.array(matrix)
    T = len(matrix)
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=[f"task_{i}" for i in range(T)],
        yticklabels=[f"after_t{i}" for i in range(T)],
    )
    plt.title(title)
    plt.xlabel("Evaluated On")
    plt.ylabel("Trained Up To")
    plt.tight_layout()
    plt.savefig(output_dir + "/" + title + "_heatmap.png")
    plt.close()


def plot_accuracy_curves(matrix, title, output_dir):
    T = len(matrix)

    plt.figure(figsize=(8, 6))
    for i in range(T):
        y_values = [matrix[t][i] for t in range(i, T)]
        x_values = list(range(i, T))
        plt.plot(x_values, y_values, marker="o", label=f"task_{i}")

    plt.title(title)
    plt.xlabel("Tasks Trained")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir + "/" + title + "_curves.png")
    plt.close()


def plot_forgetting(matrix, title, output_dir):
    T = len(matrix)
    forgetting_per_task = []

    for i in range(T - 1):
        column_i = [matrix[t][i] for t in range(i, T)]
        peak = max(column_i)
        final = matrix[T - 1][i]
        forgetting_per_task.append(peak - final)

    x_positions = list(range(T - 1))
    x_labels = [f"task_{i}" for i in range(T - 1)]

    plt.figure(figsize=(8, 6))
    plt.bar(x_positions, forgetting_per_task)
    plt.xticks(x_positions, x_labels)
    plt.xlabel("Task")
    plt.ylabel("Forgetting")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir + "/" + title + "_forgetting.png")
    plt.close()


def plot_all(method_name, class_il_matrix, task_il_matrix):
    plots_dir = "results/" + method_name + "/plots"
    os.makedirs(plots_dir, exist_ok=True)

    plot_accuracy_matrix(class_il_matrix, "class_il", plots_dir)
    plot_accuracy_matrix(task_il_matrix, "task_il", plots_dir)

    plot_accuracy_curves(class_il_matrix, "class_il", plots_dir)
    plot_accuracy_curves(task_il_matrix, "task_il", plots_dir)

    plot_forgetting(class_il_matrix, "class_il", plots_dir)
    plot_forgetting(task_il_matrix, "task_il", plots_dir)

    print(f"Plots saved in {plots_dir}")
