"""Run configured continual-learning methods from one entrypoint.

The suite builds shared tasks and model initializations, captures per-method
logs and outputs, and writes a cross-method summary under the configured
output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import benchmarker.benchmark as benchmark_module
from benchmarker.plotter import (
    plot_accuracy_curves,
    plot_accuracy_matrix,
    plot_forgetting,
)
from baselines.src.ewc import EWCMethod
from baselines.src.ewc_dr import EWCDRMethod
from baselines.src.naive import NaiveMethod
from baselines.src.si import SIMethod
from causal_coding.src.lateral import adam_init_per_layer, inv_softplus
from causal_coding.src.method import CausalCodingMethod
from causal_coding.src.model import CausalCodingModel
from causal_coding.src.vertical_pruning import (
    init_vertical_gate_logits,
    init_vertical_importance,
    vertical_adam_init,
)
from core.base import EWCState, EWCVanillaState, SIState
from core.data import load_mnist, split_into_tasks
from core.model import MLP


FAIR_METHODS = (
    "naive",
    "ewc",
    "ewc_dr",
    "online_ewc",
    "online_ewc_dr",
    "si",
    "causal_coding",
)


@dataclass
class SuiteConfig:
    dataset: str = "split_mnist"
    class_splits: list[list[int]] = field(
        default_factory=lambda: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    )
    layer_sizes: list[int] = field(default_factory=lambda: [784, 512, 512, 10])
    seed: int = 0
    batch_size: int = 128
    epochs: int = 25
    initializer: str = "he"
    init_scale: float | None = None
    output_root: str = "results/cl_suite"
    methods: list[str] = field(default_factory=lambda: list(FAIR_METHODS))


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model_kind: str
    build_method: Callable[[SuiteConfig], Any]
    build_state: Callable[[Any], Any]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_class_splits(value: str) -> list[list[int]]:
    splits = []
    for group in value.split(";"):
        classes = [item.strip() for item in group.split(",") if item.strip()]
        if not classes:
            continue
        try:
            splits.append([int(item) for item in classes])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    if not splits:
        raise argparse.ArgumentTypeError(
            "expected class splits like '0,1;2,3;4,5;6,7;8,9'"
        )
    return splits


def _parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in methods if name not in FAIR_METHODS]
    if unknown:
        valid = ", ".join(FAIR_METHODS)
        raise argparse.ArgumentTypeError(
            f"unknown method(s): {', '.join(unknown)}; valid: {valid}"
        )
    if not methods:
        raise argparse.ArgumentTypeError("expected at least one method")
    return methods


def parse_args() -> SuiteConfig:
    parser = argparse.ArgumentParser(
        description="Run the centralized continual-learning suite."
    )
    parser.add_argument("--dataset", default="split_mnist", choices=["split_mnist"])
    parser.add_argument("--class-splits", type=_parse_class_splits)
    parser.add_argument("--layer-sizes", type=_parse_int_list)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--initializer",
        default="he",
        choices=["he", "xavier", "normal", "random"],
    )
    parser.add_argument("--init-scale", type=float)
    parser.add_argument("--output-root", default="results/cl_suite")
    parser.add_argument("--methods", type=_parse_methods)
    args = parser.parse_args()

    config = SuiteConfig(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        initializer=args.initializer,
        init_scale=args.init_scale,
        output_root=args.output_root,
    )
    if args.class_splits is not None:
        config.class_splits = args.class_splits
    if args.layer_sizes is not None:
        config.layer_sizes = args.layer_sizes
    if args.methods is not None:
        config.methods = args.methods
    return config


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(_jsonable(payload), f, indent=2)


def _save_matrix_csv(output_dir: str, matrix, filename: str):
    path = os.path.join(output_dir, filename)
    header = ["after_task"] + [f"task_{i}" for i in range(len(matrix))]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t, row in enumerate(matrix):
            writer.writerow([f"after_task_{t}"] + [float(v) for v in row])


def _initializer_scale(name: str, fan_in: int, fan_out: int, scale: float | None):
    if name == "he":
        return jnp.sqrt(2.0 / fan_in)
    if name == "xavier":
        return jnp.sqrt(2.0 / (fan_in + fan_out))
    if name == "normal":
        return 0.01 if scale is None else scale
    if name == "random":
        return 0.05 if scale is None else scale
    raise ValueError(f"unknown initializer: {name}")


def _init_weight(
    key,
    shape: tuple[int, int],
    fan_in: int,
    fan_out: int,
    initializer: str,
    init_scale: float | None,
):
    scale = _initializer_scale(initializer, fan_in, fan_out, init_scale)
    if initializer == "random":
        return jax.random.uniform(key, shape, minval=-scale, maxval=scale)
    return jax.random.normal(key, shape) * scale


def _weight_keys(seed: int, num_weights: int):
    return list(jax.random.split(jax.random.PRNGKey(seed), num_weights))


def init_mlp_params(config: SuiteConfig):
    params = {}
    keys = _weight_keys(config.seed, len(config.layer_sizes) - 1)
    for i, key in enumerate(keys):
        fan_in = config.layer_sizes[i]
        fan_out = config.layer_sizes[i + 1]
        params[f"layer_{i + 1}"] = {
            "w": _init_weight(
                key,
                (fan_in, fan_out),
                fan_in,
                fan_out,
                config.initializer,
                config.init_scale,
            ),
            "b": jnp.zeros(fan_out),
        }
    return params


def init_causal_coding_params(model: CausalCodingModel, config: SuiteConfig):
    weight_keys = _weight_keys(config.seed, model.num_layers - 1)
    weights = []
    for l, key in enumerate(weight_keys):
        fan_in = model.layer_sizes[l]
        fan_out = model.layer_sizes[l + 1]
        # Initialize the same logical matrix as the MLP, then transpose for the
        # causal-coding convention `(fan_out, fan_in)`.
        w_mlp_orientation = _init_weight(
            key,
            (fan_in, fan_out),
            fan_in,
            fan_out,
            config.initializer,
            config.init_scale,
        )
        weights.append(w_mlp_orientation.T)

    log_precisions = [
        jnp.zeros(model.layer_sizes[l]) for l in range(1, model.num_layers)
    ]
    precision_var_ema = [
        jnp.ones(model.layer_sizes[l]) for l in range(1, model.num_layers - 1)
    ]

    num_hidden = model.num_layers - 2
    hidden_dims = [model.layer_sizes[l] for l in range(1, model.num_layers - 1)]
    lateral_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), 10_003)
    if num_hidden > 0:
        U_keys = jax.random.split(lateral_key, num_hidden)
        lateral_U = [
            model.lateral_init_scale
            * jax.random.normal(U_keys[i], (hidden_dims[i], model.lateral_rank))
            for i in range(num_hidden)
        ]
        rho_init_value = inv_softplus(model.alpha_init)
        lateral_log_alpha = [jnp.asarray(rho_init_value) for _ in range(num_hidden)]
        lateral_cov_ema = [jnp.eye(d) for d in hidden_dims]
        lateral_adam_states = [
            adam_init_per_layer(U, rho)
            for U, rho in zip(lateral_U, lateral_log_alpha)
        ]
        lateral_adam_step = jnp.zeros((), dtype=jnp.float32)
    else:
        lateral_U = []
        lateral_log_alpha = []
        lateral_cov_ema = []
        lateral_adam_states = []
        lateral_adam_step = jnp.zeros((), dtype=jnp.float32)

    vertical_gate_logits = init_vertical_gate_logits(weights, initial_gate=0.99)
    vertical_importance = init_vertical_importance(weights)
    vertical_adam_states = vertical_adam_init(vertical_gate_logits)
    vertical_layer_scales = [jnp.asarray(0.0) for _ in weights]
    vertical_adam_step = jnp.zeros((), dtype=jnp.float32)

    return {
        "weights": weights,
        "log_precisions": log_precisions,
        "precision_var_ema": precision_var_ema,
        "lateral_U": lateral_U,
        "lateral_log_alpha": lateral_log_alpha,
        "lateral_cov_ema": lateral_cov_ema,
        "lateral_adam_states": lateral_adam_states,
        "lateral_adam_step": lateral_adam_step,
        "vertical_gate_logits": vertical_gate_logits,
        "vertical_importance": vertical_importance,
        "vertical_adam_states": vertical_adam_states,
        "vertical_adam_step": vertical_adam_step,
        "vertical_layer_scales": vertical_layer_scales,
        "vertical_match_loss": jnp.asarray(0.0),
        "vertical_sparse_loss": jnp.asarray(0.0),
    }


def build_tasks(config: SuiteConfig):
    if config.dataset != "split_mnist":
        raise ValueError(f"unsupported dataset: {config.dataset}")
    train_X, train_y, test_X, test_y = load_mnist()
    return split_into_tasks(train_X, train_y, test_X, test_y, config.class_splits)


def _zero_tree_like(params):
    return jax.tree.map(lambda p: jnp.zeros_like(p), params)


def experiment_specs() -> dict[str, ExperimentSpec]:
    return {
        "naive": ExperimentSpec(
            name="naive",
            model_kind="mlp",
            build_method=lambda c: NaiveMethod(
                lr=0.01,
                batch_size=c.batch_size,
                epochs=c.epochs,
            ),
            build_state=lambda params: None,
        ),
        "ewc": ExperimentSpec(
            name="ewc",
            model_kind="mlp",
            build_method=lambda c: EWCMethod(
                lr=0.001,
                lr_task1=0.01,
                batch_size=c.batch_size,
                epochs=c.epochs,
                lam=1000,
                num_samples=200,
            ),
            build_state=lambda params: EWCVanillaState(anchors=[]),
        ),
        "ewc_dr": ExperimentSpec(
            name="ewc_dr",
            model_kind="mlp",
            build_method=lambda c: EWCDRMethod(
                lr=0.001,
                lr_task1=0.01,
                batch_size=c.batch_size,
                epochs=c.epochs,
                lam=100,
                num_samples=200,
            ),
            build_state=lambda params: EWCVanillaState(anchors=[]),
        ),
        "online_ewc": ExperimentSpec(
            name="online_ewc",
            model_kind="mlp",
            build_method=lambda c: EWCMethod(
                lr=0.001,
                lr_task1=0.01,
                batch_size=c.batch_size,
                epochs=c.epochs,
                lam=10000,
                num_samples=300,
                decay=0.9,
            ),
            build_state=lambda params: EWCState(
                old_params=params,
                cumulative_fisher=_zero_tree_like(params),
            ),
        ),
        "online_ewc_dr": ExperimentSpec(
            name="online_ewc_dr",
            model_kind="mlp",
            build_method=lambda c: EWCDRMethod(
                lr=0.001,
                lr_task1=0.01,
                batch_size=c.batch_size,
                epochs=c.epochs,
                lam=1000,
                num_samples=200,
                decay=0.9,
            ),
            build_state=lambda params: EWCState(
                old_params=params,
                cumulative_fisher=_zero_tree_like(params),
            ),
        ),
        "si": ExperimentSpec(
            name="si",
            model_kind="mlp",
            build_method=lambda c: SIMethod(
                lr=0.001,
                lr_task1=0.01,
                batch_size=c.batch_size,
                epochs=c.epochs,
                lam=500.0,
                normalize=True,
            ),
            build_state=lambda params: SIState(
                old_params=params,
                cumulative_omega=_zero_tree_like(params),
            ),
        ),
        "causal_coding": ExperimentSpec(
            name="causal_coding",
            model_kind="causal_coding",
            build_method=lambda c: CausalCodingMethod(
                lr_z=0.05,
                lr_w=0.05,
                num_inference_steps=30,
                gate_p=2.0,
                gate_kappa=1e-3,
                ridge=1e-4,
                lambda_s=0.0,
                batch_size=c.batch_size,
                epochs=c.epochs,
                beta_pi=0.99,
                k_probe=10,
                lr_lat=1e-3,
                beta_cov=0.99,
                eps_lat=1e-2,
                lat_warmup_epochs=5,
                lat_ramp_epochs=10,
                lambda_max_cap=1.0,
                beta_logdet=0.1,
                lambda_fro=1e-2,
                lambda_U=1e-5,
                lambda_d=1e-3,
                clarity_t=1.0,
                clarity_eps=1e-4,
                pi0=2.718281828459045,
                rho_v=0.1,
                delta_abs=1e-12,
                d_min=0.75,
                d_max=1.5,
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
            ),
            build_state=lambda params: None,
        ),
    }


def build_model_and_params(spec: ExperimentSpec, config: SuiteConfig):
    if spec.model_kind == "mlp":
        model = MLP(config.layer_sizes)
        params = init_mlp_params(config)
        return model, params
    if spec.model_kind == "causal_coding":
        model = CausalCodingModel(
            config.layer_sizes,
            lateral_rank=32,
            lateral_init_scale=1e-3,
            alpha_init=0.01,
        )
        params = init_causal_coding_params(model, config)
        return model, params
    raise ValueError(f"unsupported model kind: {spec.model_kind}")


def make_run_dir(output_root: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_root, timestamp)
    run_dir = base
    suffix = 2
    while os.path.exists(run_dir):
        run_dir = f"{base}_{suffix}"
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def patch_benchmark_outputs(method_dir: str):
    original_save_results = benchmark_module.save_results
    original_plot_all = benchmark_module.plot_all

    def suite_save_results(
        method_name,
        metrics_dict,
        class_il_matrix,
        task_il_matrix,
        config=None,
        class_il_nll_matrix=None,
        task_il_nll_matrix=None,
        task_il_bce_matrix=None,
    ):
        os.makedirs(method_dir, exist_ok=True)
        _write_json(
            os.path.join(method_dir, "metrics.json"),
            {
                "method": method_name,
                "config": config or {},
                "protocol": {
                    "task_il": metrics_dict["task_il"],
                    "class_il": metrics_dict["class_il"],
                },
            },
        )
        _save_matrix_csv(method_dir, class_il_matrix, "class_il_matrix.csv")
        _save_matrix_csv(method_dir, task_il_matrix, "task_il_matrix.csv")
        if class_il_nll_matrix is not None:
            _save_matrix_csv(method_dir, class_il_nll_matrix, "class_il_nll_matrix.csv")
        if task_il_nll_matrix is not None:
            _save_matrix_csv(method_dir, task_il_nll_matrix, "task_il_nll_matrix.csv")
        if task_il_bce_matrix is not None:
            _save_matrix_csv(method_dir, task_il_bce_matrix, "task_il_bce_matrix.csv")
        print(f"Logs saved in {method_dir}")

    def suite_plot_all(method_name, class_il_matrix, task_il_matrix):
        plots_dir = os.path.join(method_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_accuracy_matrix(class_il_matrix, "class_il", plots_dir)
        plot_accuracy_matrix(task_il_matrix, "task_il", plots_dir)
        plot_accuracy_curves(class_il_matrix, "class_il", plots_dir)
        plot_accuracy_curves(task_il_matrix, "task_il", plots_dir)
        plot_forgetting(class_il_matrix, "class_il", plots_dir)
        plot_forgetting(task_il_matrix, "task_il", plots_dir)
        print(f"Plots saved in {plots_dir}")

    benchmark_module.save_results = suite_save_results
    benchmark_module.plot_all = suite_plot_all

    @contextlib.contextmanager
    def restore_after():
        try:
            yield
        finally:
            benchmark_module.save_results = original_save_results
            benchmark_module.plot_all = original_plot_all

    return restore_after()


def _method_config(method):
    config = dict(vars(method))
    config.pop("task_il_training", None)
    return _jsonable(config)


def write_summary(run_dir: str, config: SuiteConfig, method_records: list[dict]):
    summary_rows = []
    for record in method_records:
        metrics = record["metrics"]
        task_il = metrics["task_il"]
        class_il = metrics["class_il"]
        summary_rows.append(
            {
                "method": record["method"],
                "elapsed_seconds": f"{record['elapsed_seconds']:.3f}",
                "output_dir": record["output_dir"],
                "task_il_average_accuracy": task_il["average_accuracy"],
                "task_il_average_nll": task_il["average_nll"],
                "task_il_average_bce_nll": task_il.get("average_bce_nll", ""),
                "task_il_backward_transfer": task_il["backward_transfer"],
                "task_il_forward_transfer": task_il["forward_transfer"],
                "task_il_forgetting": task_il["forgetting"],
                "class_il_average_accuracy": class_il["average_accuracy"],
                "class_il_average_nll": class_il["average_nll"],
                "class_il_backward_transfer": class_il["backward_transfer"],
                "class_il_forward_transfer": class_il["forward_transfer"],
                "class_il_forgetting": class_il["forgetting"],
            }
        )

    if not summary_rows:
        return

    summary_csv = os.path.join(run_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    _write_json(
        os.path.join(run_dir, "summary_metrics.json"),
        {
            "suite_config": asdict(config),
            "methods": method_records,
        },
    )


def run_method(
    spec: ExperimentSpec,
    config: SuiteConfig,
    tasks,
    run_dir: str,
    suite_log,
):
    method_dir = os.path.join(run_dir, spec.name)
    os.makedirs(method_dir, exist_ok=True)
    method_log_path = os.path.join(method_dir, "train.log")
    elapsed = 0.0
    results = None

    with open(method_log_path, "w") as method_log:
        out = Tee(sys.__stdout__, suite_log, method_log)
        err = Tee(sys.__stderr__, suite_log, method_log)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            print(f"\n=== Starting {spec.name} ===")
            print(f"Output directory: {method_dir}")
            start = time.time()
            model, params = build_model_and_params(spec, config)
            method = spec.build_method(config)
            state = spec.build_state(params)
            method_config = _method_config(method)
            _write_json(
                os.path.join(method_dir, "method_config.json"),
                {
                    "method": spec.name,
                    "model_kind": spec.model_kind,
                    "layer_sizes": config.layer_sizes,
                    "initializer": config.initializer,
                    "init_scale": config.init_scale,
                    "seed": config.seed,
                    "method_config": method_config,
                },
            )

            with patch_benchmark_outputs(method_dir):
                benchmark = benchmark_module.CLBenchmark(
                    method=method,
                    model=model,
                    tasks=tasks,
                    name=spec.name,
                    config=method_config,
                )
                results = benchmark.run(params, state)

            elapsed = time.time() - start
            print(f"=== Finished {spec.name} in {elapsed:.2f}s ===")

    return {
        "method": spec.name,
        "model_kind": spec.model_kind,
        "output_dir": method_dir,
        "train_log": method_log_path,
        "elapsed_seconds": elapsed,
        "config": results.config,
        "metrics": results.metrics,
        "class_il_baselines": results.class_il_baselines,
        "task_il_baselines": results.task_il_baselines,
    }


def run_suite(config: SuiteConfig):
    specs = experiment_specs()
    selected = [specs[name] for name in config.methods]
    run_dir = make_run_dir(config.output_root)
    suite_log_path = os.path.join(run_dir, "suite.log")

    _write_json(
        os.path.join(run_dir, "suite_config.json"),
        {
            "suite_config": asdict(config),
            "methods": [spec.name for spec in selected],
            "excluded_methods": ["ewc_with_ema"],
        },
    )

    method_records = []
    with open(suite_log_path, "w") as suite_log:
        out = Tee(sys.__stdout__, suite_log)
        err = Tee(sys.__stderr__, suite_log)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            print("=== Continual-learning suite ===")
            print(f"Run directory: {run_dir}")
            print(f"Methods: {', '.join(config.methods)}")
            print(f"Dataset: {config.dataset}")
            print(f"Class splits: {config.class_splits}")
            print(f"Layer sizes: {config.layer_sizes}")
            print(f"Initializer: {config.initializer}")
            print(f"Seed: {config.seed}")
            print(f"Batch size: {config.batch_size}")
            print(f"Epochs: {config.epochs}")
            print("\nLoading tasks...")
            tasks = build_tasks(config)
            print(f"Loaded {len(tasks)} tasks.")

            suite_start = time.time()
            for spec in selected:
                record = run_method(spec, config, tasks, run_dir, suite_log)
                method_records.append(record)
                write_summary(run_dir, config, method_records)

            elapsed = time.time() - suite_start
            print(f"\n=== Suite finished in {elapsed:.2f}s ===")
            print(f"Summary: {os.path.join(run_dir, 'summary.csv')}")
            print(f"Metrics: {os.path.join(run_dir, 'summary_metrics.json')}")

    return run_dir


def main():
    config = parse_args()
    run_dir = run_suite(config)
    print(f"\nSuite results saved in {run_dir}")


if __name__ == "__main__":
    main()
