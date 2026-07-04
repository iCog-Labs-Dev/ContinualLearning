"""Train the causal-coding classifier on full MNIST with diagnostics."""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.data import load_mnist, Task
from src.diagnostics import pprint_diag_summary, run_full_diagnostics
from src.method import CausalCodingMethod
from src.model import CausalCodingModel


DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "cc_full_mnist"
)
RESULTS_DIR = os.environ.get("CC_RESULTS_DIR", DEFAULT_RESULTS_DIR)
DIAG_BATCH_SIZE = 256
INCLUDE_VLCP_SWEEP = False


def _dump_params_for_eigenanalysis(params, path):
    """Save trained arrays needed for the Λ/C_ema eigenanalysis.

    Skips Adam states, gate logits, and importance arrays — the eigenanalysis
    only needs the model matrices.
    """
    out = {}
    for l, w in enumerate(params.get("weights", [])):
        out[f"W_{l}"] = np.asarray(w)
    for l, lp in enumerate(params.get("log_precisions", [])):
        out[f"log_pi_{l}"] = np.asarray(lp)
    for l, U in enumerate(params.get("lateral_U", [])):
        out[f"lateral_U_{l}"] = np.asarray(U)
    for l, rho in enumerate(params.get("lateral_log_alpha", [])):
        out[f"lateral_rho_{l}"] = np.asarray(rho)
    for l, C in enumerate(params.get("lateral_cov_ema", [])):
        out[f"lateral_cov_ema_{l}"] = np.asarray(C)
    np.savez_compressed(path, **out)


def _vertical_mode_config():
    mode = os.environ.get("VLCP_VERTICAL_MODE", "baseline").strip().lower()
    mode_table = {
        "baseline": (False, (1.0, 1.0, 0.0)),
        "hidden-only": (True, (1.0, 1.0, 0.0)),
        "soft-output": (True, (1.0, 1.0, 0.25)),
        "full": (True, (1.0, 1.0, 1.0)),
        "full-all-layer": (True, (1.0, 1.0, 1.0)),
    }
    if mode not in mode_table:
        valid = ", ".join(sorted(mode_table))
        raise ValueError(f"Unknown VLCP_VERTICAL_MODE={mode!r}; expected one of {valid}")
    enabled, layer_scales = mode_table[mode]
    return mode, enabled, layer_scales


def _run_config(model, method, vertical_mode):
    return {
        "vertical_mode": vertical_mode,
        "layer_sizes": model.layer_sizes,
        "lateral_rank": model.lateral_rank,
        "lateral_init_scale": model.lateral_init_scale,
        "alpha_init": model.alpha_init,
        "lr_z": method.lr_z,
        "lr_w": method.lr_w,
        "num_inference_steps": method.num_inference_steps,
        "gate_p": method.gate_p,
        "gate_kappa": method.gate_kappa,
        "ridge": method.ridge,
        "lambda_s": method.lambda_s,
        "batch_size": method.batch_size,
        "epochs": method.epochs,
        "beta_pi": method.beta_pi,
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
        # Diffusion clarity penalty config.
        "lambda_d": method.lambda_d,
        "clarity_t": method.clarity_t,
        "clarity_eps": method.clarity_eps,
        # Structured diagonal residual precision config.
        "pi0": method.pi0,
        "rho_v": method.rho_v,
        "delta_abs": method.delta_abs,
        "d_min": method.d_min,
        "d_max": method.d_max,
        # Vertical soft-pruning gates.
        "vertical_pruning_enabled": method.vertical_pruning_enabled,
        "vertical_layer_scales": list(method.vertical_layer_scales),
        "vertical_alpha_g": method.vertical_alpha_g,
        "lambda_vert_match": method.lambda_vert_match,
        "lambda_vert_sparse": method.lambda_vert_sparse,
        "vertical_eps": method.vertical_eps,
        "lr_vert": method.lr_vert,
        "vertical_warmup_epochs": method.vertical_warmup_epochs,
        "vertical_ramp_epochs": method.vertical_ramp_epochs,
        "vertical_importance_update_epochs": method.vertical_importance_update_epochs,
        "vertical_importance_batch_size": method.vertical_importance_batch_size,
        "vertical_prune_threshold": method.vertical_prune_threshold,
        "vertical_importance_lambda_dyn": method.vertical_importance_lambda_dyn,
        "vertical_importance_window": method.vertical_importance_window,
        "vertical_importance_num_iters": method.vertical_importance_num_iters,
        "vertical_importance_standardize_x": (
            method.vertical_importance_standardize_x
        ),
        "diag_batch_size": DIAG_BATCH_SIZE,
        "include_vlcp_sweep": INCLUDE_VLCP_SWEEP,
    }


def _split_vlcp_records(records):
    main_records = []
    vlcp_records = []

    for record in records:
        main_record = dict(record)
        vlcp_payload = main_record.pop("vlcp_regression", None)
        main_records.append(main_record)

        if vlcp_payload is not None:
            vlcp_records.append(
                {
                    "label": record.get("label"),
                    "performance": record.get("performance"),
                    "vlcp_regression": vlcp_payload,
                }
            )

    assert all("vlcp_regression" not in r for r in main_records)
    assert all("vlcp_regression" in r for r in vlcp_records)
    return main_records, vlcp_records


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
    vertical_mode, vertical_enabled, vertical_layer_scales = _vertical_mode_config()
    print(
        "VLCP vertical mode: "
        f"{vertical_mode} "
        f"(enabled={vertical_enabled}, layer_scales={vertical_layer_scales})"
    )

    # Full-MNIST settings: tight structured Pi, stable lateral settings,
    # clarity off, and configurable vertical pruning.
    method_kwargs = dict(
        lr_z=0.05,
        lr_w=0.02,
        num_inference_steps=30,
        gate_p=2.0,
        gate_kappa=1e-3,
        ridge=1e-4,
        lambda_s=1e-6,
        batch_size=128,
        epochs=50,
        beta_pi=0.99,
        k_probe=10,
        # Lateral precision settings.
        lr_lat=1e-3,
        beta_cov=0.99,
        eps_lat=1e-2,
        lat_warmup_epochs=5,
        lat_ramp_epochs=10,
        lambda_max_cap=1.0,
        beta_logdet=0.1,
        lambda_fro=1e-2,
        lambda_U=1e-5,
        # Clarity disabled for this run.
        lambda_d=0.0,
        clarity_t=1.0,
        clarity_eps=1e-4,
        # Tight-clipped structured diagonal residual precision.
        pi0=2.718281828459045,
        rho_v=0.1,
        delta_abs=1e-12,
        d_min=0.75,
        d_max=1.5,
        # Vertical soft-pruning gates.
        vertical_pruning_enabled=vertical_enabled,
        vertical_layer_scales=vertical_layer_scales,
    )
    method = CausalCodingMethod(**method_kwargs)

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
        include_vlcp_sweep=INCLUDE_VLCP_SWEEP,
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
    config = _run_config(model, method, vertical_mode)
    main_records, vlcp_records = _split_vlcp_records(diag_records)

    output_path = os.path.join(RESULTS_DIR, "diagnostics.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": config,
                "records": main_records,
            },
            f,
            indent=2,
        )
    print(f"\nDiagnostics written to {output_path}")

    vlcp_output_path = os.path.join(RESULTS_DIR, "vlcp_regression_diagnostics.json")
    with open(vlcp_output_path, "w") as f:
        json.dump(
            {
                "config": {
                    **config,
                    "source_diagnostics_file": os.path.basename(output_path),
                },
                "records": vlcp_records,
            },
            f,
            indent=2,
        )
    print(f"VLCP regression diagnostics written to {vlcp_output_path}")

    # Dump trained model matrices for later eigenanalysis
    # (Λ, C_ema, weights).
    params_path = os.path.join(RESULTS_DIR, "params.npz")
    _dump_params_for_eigenanalysis(params, params_path)
    print(f"Params written to {params_path}")


if __name__ == "__main__":
    main()
