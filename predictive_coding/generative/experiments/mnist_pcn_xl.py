"""
MNIST Baseline Experiment — forward_xl evaluation

Train the generative PCN on full MNIST (all 10 classes, no task splits).
Evaluation uses forward_xl: single inference pass, settled top-layer as logits.
Cost: 1 × B × T_infer  (vs 10 × B × T_infer for the exhaustive forward).
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import GenerativePCN
from src.training import train_step


def load_mnist():
    """Load and preprocess MNIST via Keras. Returns JAX arrays."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Flatten and normalise to [-1, 1]
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 127.5 - 1.0
    x_test  = x_test.reshape(-1, 784).astype(np.float32) / 127.5 - 1.0

    y_train_oh = np.eye(10, dtype=np.float32)[y_train]
    y_test_oh  = np.eye(10, dtype=np.float32)[y_test]

    return (
        jnp.array(x_train), jnp.array(y_train), jnp.array(y_train_oh),
        jnp.array(x_test),  jnp.array(y_test),  jnp.array(y_test_oh),
    )


def make_batches(X, Y, Y_idx, batch_size, key):
    """Shuffle and yield (x_batch, y_oh_batch, y_idx_batch) tuples."""
    n = X.shape[0]
    perm = jax.random.permutation(key, n)
    X, Y, Y_idx = X[perm], Y[perm], Y_idx[perm]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[start:end], Y[start:end], Y_idx[start:end]


def evaluate(pcn, params, X, Y_idx, batch_size=512, inference_steps=40):
    """Accuracy via forward_xl: single pass, settled top-layer as logits."""
    n = X.shape[0]
    correct = 0
    for start in range(0, n, batch_size):
        end    = min(start + batch_size, n)
        logits = pcn.forward_xl(params, X[start:end], inference_steps=inference_steps)
        preds  = jnp.argmax(logits, axis=1)
        correct += int(jnp.sum(preds == Y_idx[start:end]))
    return correct / n


def main():
    print("=" * 60)
    print("  PCN MNIST Baseline — forward_xl (single-pass top-layer)")
    print("=" * 60)

    # Hyperparameters (same as mnist_pcn.py)
    LAYER_SIZES     = [784, 512, 512, 10]
    EPOCHS          = 35
    BATCH_SIZE      = 512
    ETA_X           = 0.1
    ETA_W           = 0.005
    INFERENCE_STEPS = 50
    SEED            = 42
    EVAL_EVERY      = 5

    print(f"\n  Architecture : {LAYER_SIZES}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  eta_x        : {ETA_X}")
    print(f"  eta_w        : {ETA_W}")
    print(f"  Inf. steps   : {INFERENCE_STEPS}")
    print(f"  Evaluation   : forward_xl (1 pass, free top layer)")
    print()

    print("Loading MNIST...")
    x_train, y_train, y_train_oh, x_test, y_test, y_test_oh = load_mnist()
    print(f"  Train: {x_train.shape}  Test: {x_test.shape}\n")

    pcn    = GenerativePCN(layer_sizes=LAYER_SIZES)
    key    = jax.random.PRNGKey(SEED)
    params = pcn.init_params(key)

    num_batches = int(np.ceil(x_train.shape[0] / BATCH_SIZE))

    print(f"{'Epoch':>6}  {'Train E':>10}  {'Train Acc':>10}  {'Test Acc':>10}  {'Time(s)':>8}", flush=True)
    print("-" * 56, flush=True)

    for epoch in range(EPOCHS):
        t0 = time.time()
        key, subkey = jax.random.split(key)

        epoch_energy = 0.0
        for xb, yb, _ in make_batches(x_train, y_train_oh, y_train, BATCH_SIZE, subkey):
            params, metrics = train_step(
                params, xb, yb,
                eta_x=ETA_X,
                eta_w=ETA_W,
                inference_steps=INFERENCE_STEPS,
                init_mode="bottom_up",
            )
            epoch_energy += float(metrics["energy"])

        avg_energy = epoch_energy / num_batches

        train_acc = evaluate(pcn, params, x_train, y_train,
                             batch_size=512, inference_steps=INFERENCE_STEPS)

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0 or epoch == EPOCHS - 1:
            test_acc = evaluate(pcn, params, x_test, y_test,
                                batch_size=512, inference_steps=INFERENCE_STEPS)
            test_str = f"{test_acc * 100:>9.2f}%"
        else:
            test_str = "        -"

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {avg_energy:>10.4f}  {train_acc * 100:>9.2f}%  {test_str}  {elapsed:>8.1f}s", flush=True)

    print("\n" + "=" * 60)
    final_train = evaluate(pcn, params, x_train, y_train, batch_size=512,
                           inference_steps=INFERENCE_STEPS)
    final_test  = evaluate(pcn, params, x_test,  y_test,  batch_size=512,
                           inference_steps=INFERENCE_STEPS)
    print(f"  Final Train Accuracy : {final_train * 100:.2f}%")
    print(f"  Final Test  Accuracy : {final_test  * 100:.2f}%")
    print("=" * 60)

    if final_test < 0.50:
        print("\n  [WARN] Test accuracy < 50% — model may not have learned.")
    elif final_test < 0.80:
        print("\n  [OK]   Model is learning but not yet competitive.")
    else:
        print("\n  [PASS] Reasonable MNIST accuracy — forward_xl is working.")


if __name__ == "__main__":
    main()
