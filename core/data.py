from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.keras.datasets import mnist


@dataclass
class Task:
    train_X: jnp.ndarray
    train_y: jnp.ndarray
    test_X: jnp.ndarray
    test_y: jnp.ndarray
    classes: list


def load_mnist():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
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
