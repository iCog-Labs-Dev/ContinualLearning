"""
Discriminative PCN diagnostics.

Run with:
    python -m predictive_coding.experiments.diagnose_discriminative
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.model_discriminative import DiscriminativePCN
from src.training_discriminative import train_step_discriminative
from src.inference_discriminative import settle_states_discriminative
from src.energy_discriminative import compute_errors_discriminative


def separator(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "-" * pad + f" {title} " + "-" * pad)
    else:
        print("\n" + "-" * w)


def generate_synthetic_data(num_samples=100, num_classes=10, feature_dim=784, seed=42):
    key = jax.random.PRNGKey(seed)
    key_proto, key_noise = jax.random.split(key)
    prototypes = jnp.tanh(jax.random.normal(key_proto, (num_classes, feature_dim)))
    labels = jnp.tile(jnp.arange(num_classes), num_samples // num_classes)
    noise = 0.1 * jax.random.normal(key_noise, (num_samples, feature_dim))
    data = jnp.tanh(prototypes[labels] + noise)
    labels_onehot = jax.nn.one_hot(labels, num_classes)
    return data, labels, labels_onehot


def quick_train(params, X, Y, epochs=200, eta_x=0.1, eta_w=0.01, inference_steps=50):
    for epoch in range(epochs):
        params, metrics = train_step_discriminative(
            params, X, Y, eta_x=eta_x, eta_w=eta_w, inference_steps=inference_steps,
        )
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  |  energy {metrics['energy']:.4f}")
    return params


def check_settling_plateau(params, X, Y, num_steps=50, eta_x=0.1):
    separator("Settling plateau check")
    _, energy_hist = settle_states_discriminative(
        params, X, Y, num_steps=num_steps, eta_x=eta_x, mode="bottom_up",
        clamped_indices=frozenset({0, len(params)}),
    )
    e_hist = np.array(energy_hist)
    checkpoints = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    for i in checkpoints:
        print(f"  step {i:3d}: {e_hist[i]:.4f}")
    delta = e_hist[3 * num_steps // 4] - e_hist[num_steps - 1]
    if delta < 1e-4:
        print(f"  [WARN] Flat in last quarter (delta={delta:.6f}) — possible plateau.")
    else:
        print(f"  [OK]   Still decreasing (delta={delta:.4f}).")


def check_per_layer_energy(params, X, Y, num_steps=50, eta_x=0.1):
    separator("Per-layer energy check")
    states, _ = settle_states_discriminative(
        params, X, Y, num_steps=num_steps, eta_x=eta_x, mode="bottom_up",
        clamped_indices=frozenset({0, len(params)}),
    )
    errors = compute_errors_discriminative(params, states)
    for i in sorted(errors.keys()):
        e = float(jnp.mean(jnp.sum(errors[i] ** 2, axis=-1)))
        print(f"  layer -> state {i}: mean sq error = {e:.4f}")


def check_settled_vs_single_pass(pcn, params, X, Y_idx, num_steps=40, eta_x=0.1):
    separator("Settled vs single-pass accuracy")
    logits_settled = pcn.forward(params, X, inference_steps=num_steps, eta_x=eta_x)
    preds_settled = jnp.argmax(logits_settled, axis=1)
    acc_settled = float(jnp.mean(preds_settled == Y_idx))

    logits_single = pcn.forward_single_pass(params, X)
    preds_single = jnp.argmax(logits_single, axis=1)
    acc_single = float(jnp.mean(preds_single == Y_idx))

    print(f"  settled accuracy     : {acc_settled * 100:.2f}%")
    print(f"  single-pass accuracy : {acc_single * 100:.2f}%")
    if acc_settled - acc_single < 0.005:
        print("  [WARN] Settling adds negligible accuracy over a single pass.")
    else:
        print("  [OK]   Settling meaningfully improves over single pass.")


def main():
    separator("DISCRIMINATIVE PCN DIAGNOSTICS")

    X, Y_idx, Y = generate_synthetic_data()
    print(f"Data: {X.shape}  Labels: {Y.shape}")

    pcn = DiscriminativePCN(layer_sizes=[784, 256, 256, 10], activation="tanh")
    key = jax.random.PRNGKey(0)
    params = pcn.init_params(key)

    separator("Training 200 epochs (eta_w=0.01)")
    params = quick_train(params, X, Y, epochs=200, eta_w=0.01)

    check_settling_plateau(params, X, Y)
    check_per_layer_energy(params, X, Y)
    check_settled_vs_single_pass(pcn, params, X, Y_idx)

    separator("DONE")


if __name__ == "__main__":
    main()
