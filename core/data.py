import os
import urllib.request
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Task:
    train_X: jnp.ndarray
    train_y: jnp.ndarray
    test_X: jnp.ndarray
    test_y: jnp.ndarray
    classes: list


def load_mnist():
    cache_dir = os.path.expanduser("~/.keras/datasets")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "mnist.npz")
    if not os.path.exists(cache_path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"Downloading MNIST dataset to {cache_path}...")
        urllib.request.urlretrieve(url, cache_path)

    with np.load(cache_path, allow_pickle=True) as f:
        train_X, train_y = f["x_train"], f["y_train"]
        test_X, test_y = f["x_test"], f["y_test"]

    train_X = train_X.reshape(-1, 784) / 255.0
    test_X = test_X.reshape(-1, 784) / 255.0

    return train_X, train_y, test_X, test_y


def split_into_tasks(train_X, train_y, test_X, test_y, class_pairs):
    tasks = []

    for pairs in class_pairs:
        train_mask = False
        for c in pairs:
            train_mask = train_mask | (train_y == c)

        cur_train_X = train_X[train_mask]
        cur_train_y = train_y[train_mask]

        test_mask = False
        for c in pairs:
            test_mask = test_mask | (test_y == c)

        cur_test_X = test_X[test_mask]
        cur_test_y = test_y[test_mask]

        tasks.append(
            Task(
                train_X=jnp.array(cur_train_X),
                train_y=jnp.array(cur_train_y),
                test_X=jnp.array(cur_test_X),
                test_y=jnp.array(cur_test_y),
                classes=pairs,
            )
        )

    return tasks
