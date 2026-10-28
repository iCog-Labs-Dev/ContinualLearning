"""Causal-coding continual-learning benchmark with the structured lateral matrix.

Same as run_causal_coding.py, plus an --arm switch:

  L1  current low-rank lateral          (identical to the original script)
  S0  paper's V, fixed constants        (dynamics only)
  S1  V with two learned scalars/layer
  S2  V with per-unit s_i and per-pair b_ij

Examples (from the repo root):
  python causal_coding/experiments/run_causal_coding_struct.py --arm L1
  python causal_coding/experiments/run_causal_coding_struct.py --arm S0
  python causal_coding/experiments/run_causal_coding_struct.py --arm S1 --rho 0.15
  python causal_coding/experiments/run_causal_coding_struct.py --arm S1 --freeze-after-task1
  python causal_coding/experiments/run_causal_coding_struct.py --arm S1 --schur-include
"""
import argparse
import os
import sys

import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.data import load_mnist, split_into_tasks
from benchmarker import CLBenchmark
from src.model import CausalCodingModel
from src.method import CausalCodingMethod

ARMS = {
    "L0": ("off", "fixed"),
    "L1": ("lowrank", "fixed"),
    "S0": ("structured", "fixed"),
    "S1": ("structured", "scalar"),
    "S2": ("structured", "perunit"),
    "B0": ("both", "fixed"),
    "B1": ("both", "scalar"),
    "B2": ("both", "perunit"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=sorted(ARMS), default="L1")
ap.add_argument("--rho", type=float, default=0.2,
                help="target active fraction for S1/S2 (take it from run_struct_lateral.py's "
                     "active_frac_without_V; 0.2 is a placeholder)")
ap.add_argument("--group-size", type=int, default=16)
ap.add_argument("--alpha-e", type=float, default=0.13)
ap.add_argument("--inhib-total", type=float, default=0.25)
ap.add_argument("--lr-s", type=float, default=1e-3)
ap.add_argument("--lr-b", type=float, default=1e-3)
ap.add_argument("--guard-c", type=float, default=0.5)
ap.add_argument("--schur-include", action="store_true",
                help="put the structured matrix into the Schur/gate step")
ap.add_argument("--freeze-after-task1", action="store_true",
                help="stop learning s, b after the first task")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr-w", type=float, default=0.05, help="Hebbian weight learning rate")
ap.add_argument("--lambda-s", type=float, default=0.0, help="weight decay / regularization coefficient")
ap.add_argument("--gate-norm", choices=["sum", "match"], default="sum",
                help="causal gate normalization: 'sum' (default) or 'match' (preserves update norm)")
ap.add_argument("--no-vertical", action="store_true",
                help="disable vertical pruning")
ap.add_argument("--seed", type=int, default=0,
                help="random seed for initialization and training")
ap.add_argument("--task-1-only", action="store_true",
                help="run only the first task for fast diagnostic sweeps")
args = ap.parse_args()

lateral_mode, struct_learning = ARMS[args.arm]
run_name = "causal_coding" if args.arm == "L1" and not args.schur_include else (
    f"causal_coding_{args.arm}"
    + ("_schur" if args.schur_include else "")
    + ("_freeze" if args.freeze_after_task1 else "")
)
if args.seed != 0:
    run_name += f"_seed{args.seed}"
if args.no_vertical:
    run_name += "_noVert"
if args.gate_norm != "sum":
    run_name += f"_{args.gate_norm}"
if args.lambda_s == 0.0:
    run_name += "_noDecay"

X, y, test_X, test_y = load_mnist()
class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)
if args.task_1_only:
    tasks = tasks[:1]

model = CausalCodingModel(
    [784, 512, 512, 10],
    lateral_rank=32,
    lateral_init_scale=1e-3,
    alpha_init=0.01,
    # structured lateral (ignored when lateral_mode="lowrank")
    lateral_mode=lateral_mode,
    struct_group_size=args.group_size,
    struct_alpha_e=args.alpha_e,
    struct_inhib_total=args.inhib_total,
    struct_rho=args.rho,
)
key = jax.random.PRNGKey(args.seed)
params = model.init_params(key)

method = CausalCodingMethod(
    lr_z=0.05,
    lr_w=args.lr_w,
    num_inference_steps=30,
    gate_p=2.0,
    gate_kappa=1e-3,
    ridge=1e-4,
    lambda_s=args.lambda_s,
    batch_size=128,
    epochs=args.epochs,
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

    # Tight-clipped structured diagonal residual precision.
    pi0=2.718281828459045,
    rho_v=0.1,
    delta_abs=1e-12,
    d_min=0.75,
    d_max=1.5,

    # Soft hidden-only vertical pruning gates.
    vertical_pruning_enabled=not args.no_vertical,
    gate_norm=args.gate_norm,
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

    # Structured lateral matrix (rho / group size are read from the model).
    lateral_mode=lateral_mode,
    struct_learning=struct_learning,
    schur_include_structured=args.schur_include,
    struct_lr_s=args.lr_s,
    struct_lr_b=args.lr_b,
    struct_beta_stat=0.99,
    struct_guard_c=args.guard_c,
    struct_freeze_after_task1=args.freeze_after_task1,
)

# Record the model-side settings too (CLBenchmark only saves vars(method) by default).
config = dict(vars(method))
config.update(
    arm=args.arm,
    seed=args.seed,
    struct_rho=args.rho,
    struct_group_size=args.group_size,
    struct_alpha_e=args.alpha_e,
    struct_inhib_total=args.inhib_total,
)

CLBenchmark(method=method, model=model, tasks=tasks, name=run_name, config=config).run(
    params, None
)