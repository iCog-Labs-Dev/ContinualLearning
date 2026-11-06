import argparse
import sys
import os

# Configure JAX XLA memory allocation before jax import to avoid cuSolver handle exhaustion
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import json
import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.data import load_mnist, split_into_tasks
from benchmarker import CLBenchmark
from src.model import CausalCodingModel
from src.method import CausalCodingMethod


def main():
    parser = argparse.ArgumentParser(description="Run Meta Causal Coding on Split-MNIST")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (default: meta_causal_coding or meta_causal_coding_seed<seed>)")
    parser.add_argument("--fixed-controller", action="store_true", help="Freeze meta-controller parameters at initial values (ablation baseline)")
    parser.add_argument("--disable-exploration", action="store_true", help="Disable UCB exploration bonus by enforcing eps_exp=0 (ablation baseline)")
    args = parser.parse_args()

    seed = args.seed
    if args.name is not None:
        exp_name = args.name
    else:
        parts = ["meta_causal_coding"]
        if args.fixed_controller:
            parts.append("fixed_ctrl")
        if args.disable_exploration:
            parts.append("no_exp")
        if seed != 0:
            parts.append(f"seed{seed}")
        exp_name = "_".join(parts)

    print(f"Starting Meta Causal Coding experiment: name='{exp_name}', seed={seed}")

    # 1. Load Split-MNIST dataset
    X, y, test_X, test_y = load_mnist()
    class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)

    # 2. Instantiate CausalCodingModel
    model = CausalCodingModel(
        [784, 512, 512, 10],
        lateral_rank=32,
        lateral_init_scale=1e-3,
        alpha_init=0.01,
    )
    key = jax.random.PRNGKey(seed)
    params = model.init_params(key)

    # 3. Instantiate CausalCodingMethod with meta_learning_enabled=True
    method = CausalCodingMethod(
        lr_z=0.05,
        lr_w=0.05,
        num_inference_steps=30,
        gate_p=2.0,
        gate_kappa=1e-3,
        ridge=1e-4,
        lambda_s=0.0,
        batch_size=128,
        epochs=60,
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

        # Lateral clarity enabled.
        lambda_d=1e-3,
        clarity_t=1.0,
        clarity_eps=1e-4,

        # Structured diagonal residual precision.
        pi0=2.718281828459045,
        rho_v=0.1,
        delta_abs=1e-12,
        d_min=0.75,
        d_max=1.5,

        # Soft hidden-only vertical pruning gates.
        vertical_pruning_enabled=True,
        vertical_layer_scales=(1.0, 1.0, 0.0),
        vertical_alpha_g=8.0,
        lambda_vert_match=1e-3,
        lambda_vert_sparse=1e-5,
        vertical_eps=1e-2,
        lr_vert=3e-3,
        vertical_warmup_epochs=30,
        vertical_ramp_epochs=15,
        vertical_importance_update_epochs=5,
        vertical_importance_batch_size=256,
        vertical_prune_threshold=0.2,
        vertical_importance_lambda_dyn=1e-3,
        vertical_importance_window=15,
        vertical_importance_num_iters=50,
        vertical_importance_standardize_x=False,

        # Meta-Learning extension (Goertzel 2025)
        meta_learning_enabled=True,
        lr_meta=1e-2,
        meta_beta_lcb_init=1.0,
        meta_tau_init=1e-3,
        meta_eps_exp_init=0.05,
        meta_protection_init=0.8,
        anchor_size_per_task=64,
        meta_alpha_forget=1.0,
        meta_beta_confusion=0.5,
        meta_calibrate_every_epochs=5,
        fixed_controller=args.fixed_controller,
        disable_exploration=args.disable_exploration,
    )

    # 4. Run Benchmark Suite
    benchmark = CLBenchmark(method=method, model=model, tasks=tasks, name=exp_name)
    results = benchmark.run(params, None)

    # 5. Save meta diagnostics
    out_dir = os.path.join("results", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    diag_path = os.path.join(out_dir, "meta_diagnostics.json")

    diagnostics_data = getattr(method, "meta_diagnostics", None)
    if not diagnostics_data:
        meta_state = getattr(method, "last_meta_state", None)
        diagnostics_data = {
            "trajectory": meta_state.meta_trajectory if meta_state else [],
            "drift_history": meta_state.drift_history if meta_state else [],
            "confusion_graph": meta_state.confusion_graph if meta_state else [],
        }

    def _json_serial(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "__float__"):
            return float(obj)
        if hasattr(obj, "__int__"):
            return int(obj)
        return str(obj)

    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics_data, f, indent=2, default=_json_serial)

    print(f"Meta-diagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
