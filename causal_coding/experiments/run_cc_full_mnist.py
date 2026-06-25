"""Train the causal-coding classifier on full MNIST with diagnostics."""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.data import load_mnist, Task
from src.diagnostics import pprint_diag_summary, run_full_diagnostics
from src.method import CausalCodingMethod
from src.model import CausalCodingModel


RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "cc_full_mnist"
)
DIAG_BATCH_SIZE = 256


def main():
    print("Loading MNIST...")
    train_X, train_y, test_X, test_y = load_mnist()
    train_X = jnp.array(train_X)
    train_y = jnp.array(train_y)
    test_X = jnp.array(test_X)
    test_y = jnp.array(test_y)

    # Use one fixed sample for comparable diagnostics across epochs.
    diag_X = test_X[:DIAG_BATCH_SIZE]
    diag_y = test_y[:DIAG_BATCH_SIZE]

    task = Task(
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        classes=list(range(10)),
    )

    print("Building model and method...")
    model = CausalCodingModel(
        [784, 512, 512, 10],
        lateral_rank=32,
        lateral_init_scale=1e-3,
        alpha_init=0.01,
    )
    params = model.init_params(jax.random.PRNGKey(0))

    method = CausalCodingMethod(
        lr_z=0.05,
        lr_w=0.02,
        num_inference_steps=30,
        gate_p=2.0,
        gate_kappa=1e-3,
        ridge=1e-4,
        lambda_s=1e-6,
        batch_size=128,
        epochs=50,
        gate_alpha=0.3,
        gate_floor_coeff=0.05,
        gate_warmup_epochs=15,
        gate_ramp_epochs=15,
        beta_pi=0.99,
        epsilon_pi=0.1,
        alpha_pi=0.05,
        k_probe=10,
        # Lateral precision optimization.
        lr_lat=1e-3,
        beta_cov=0.99,
        eps_lat=1e-2,
        lat_warmup_epochs=5,
        lat_ramp_epochs=10,
        lambda_max_cap=1.0,
        beta_logdet=0.1,
        lambda_fro=1e-2,
        lambda_U=1e-5,
    )

    diag_records = []

    # Full diagnostics before training.
    print("\n=== Pre-training diagnostics ===")
    pre_diag = run_full_diagnostics(
        model, params, method,
        diag_X, diag_y,
        train_X, train_y,
        test_X, test_y,
        label="pre",
        include_structural=True,
    )
    pprint_diag_summary(pre_diag)
    diag_records.append(pre_diag)

    # Per-epoch diagnostics skip the more expensive state-comparison diagnostics.
    def epoch_hook(epoch_params, epoch_idx, epoch_loss):
        label = f"epoch_{epoch_idx + 1}"
        diag = run_full_diagnostics(
            model, epoch_params, method,
            diag_X, diag_y,
            train_X, train_y,
            test_X, test_y,
            label=label,
            include_structural=False,
        )
        diag["loss_clamped"] = float(epoch_loss)
        diag_records.append(diag)
        pprint_diag_summary(diag, header=f"-- {label} --")

    # Train with per-epoch diagnostics.
    print("\nTraining on full MNIST (10 classes, single task)...")
    t0 = time.time()
    params, _, final_loss = method.train_task(
        model, params, None, task, task_idx=0, diagnostics_hook=epoch_hook
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s.  Final epoch CE: {final_loss:.4f}")

    # Full diagnostics after training.
    print("\n=== Post-training diagnostics ===")
    post_diag = run_full_diagnostics(
        model, params, method,
        diag_X, diag_y,
        train_X, train_y,
        test_X, test_y,
        label="post",
        include_structural=True,
    )
    pprint_diag_summary(post_diag)
    diag_records.append(post_diag)

    # Headline summary.
    p = post_diag["performance"]
    print("\n=== Full MNIST causal-coding run ===")
    print(f"Test accuracy:  {p['test_acc'] * 100:.2f}%")
    print(f"Train accuracy: {p['train_acc'] * 100:.2f}%")
    print(f"Test NLL:       {p['test_nll']:.4f}")

    # Persist diagnostics and the run configuration.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "diagnostics.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": {
                    "layer_sizes": model.layer_sizes,
                    "lateral_rank": model.lateral_rank,
                    "lateral_init_scale": model.lateral_init_scale,
                    "alpha_init": model.alpha_init,
                    "lr_z": method.lr_z,
                    "lr_w": method.lr_w,
                    "num_inference_steps": method.num_inference_steps,
                    "gate_p": method.gate_p,
                    "gate_kappa": method.gate_kappa,
                    "gate_alpha": method.gate_alpha,
                    "gate_floor_coeff": method.gate_floor_coeff,
                    "gate_warmup_epochs": method.gate_warmup_epochs,
                    "gate_ramp_epochs": method.gate_ramp_epochs,
                    "ridge": method.ridge,
                    "lambda_s": method.lambda_s,
                    "batch_size": method.batch_size,
                    "epochs": method.epochs,
                    "beta_pi": method.beta_pi,
                    "epsilon_pi": method.epsilon_pi,
                    "alpha_pi": method.alpha_pi,
                    "k_probe": method.k_probe,
                    # Lateral precision optimization.
                    "lr_lat": method.lr_lat,
                    "beta_cov": method.beta_cov,
                    "eps_lat": method.eps_lat,
                    "lat_warmup_epochs": method.lat_warmup_epochs,
                    "lat_ramp_epochs": method.lat_ramp_epochs,
                    "lambda_max_cap": method.lambda_max_cap,
                    "beta_logdet": method.beta_logdet,
                    "lambda_fro": method.lambda_fro,
                    "lambda_U": method.lambda_U,
                    "adam_beta1_lat": method.adam_beta1_lat,
                    "adam_beta2_lat": method.adam_beta2_lat,
                    "adam_eps_lat": method.adam_eps_lat,
                    "grad_clip_norm_lat": method.grad_clip_norm_lat,
                    "diag_batch_size": DIAG_BATCH_SIZE,
                },
                "records": diag_records,
            },
            f,
            indent=2,
        )
    print(f"\nDiagnostics written to {output_path}")


if __name__ == "__main__":
    main()
