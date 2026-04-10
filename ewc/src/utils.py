import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def he_init(key, fan_in, fan_out):
    return jax.random.normal(key, shape=(fan_in, fan_out)) * jnp.sqrt(2 / fan_in)


def cross_entropy(logits, labels):
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def accuracy(predictions, labels):
    return jnp.mean(predictions == labels)


def average_accuracy(matrix):
    return jnp.mean(jnp.array(matrix)[-1])


def plot_accuracy_matrix(matrix, title, save_path):
    data = np.array(matrix)
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"],
        yticklabels=["After T1", "After T2", "After T3", "After T4", "After T5"],
    )
    plt.title(title)
    plt.xlabel("Evaluated On")
    plt.ylabel("Trained Up To")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
