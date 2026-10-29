"""Structured lateral matrix experiment runner (one arm per invocation).

Arms
----
  L1  current low-rank lateral (lateral_mode="lowrank")       - comparison baseline
  S0  paper's V, fixed constants, dynamics only               - control for learning
  S1  V with two learned scalars per layer (s, b)             - struct_learning="scalar"
  S2  V with per-unit s_i and per-pair b_ij                   - struct_learning="perunit"

Add --schur-include to also put the structured matrix into the Schur / gate
computation (Stage F; only meaningful for the S arms).

Usage (from repo root)
----------------------
  python causal_coding/experiments/run_struct_lateral.py --arm S0
  python causal_coding/experiments/run_struct_lateral.py --arm L1
  python causal_coding/experiments/run_struct_lateral.py --arm S1 --rho 0.15
  python causal_coding/experiments/run_struct_lateral.py --arm S0 --smoke   # 1-minute self-test

Recommended order: L1 and S0 first.  Compare `active_frac_with_V` against
`active_frac_without_V` in the log: if V does not lower the active fraction,
stop before S1 / S2.  Use the measured `active_frac_without_V` to choose --rho.

The runner does NOT call src.diagnostics.run_full_diagnostics: that module
rebuilds plain low-rank pairs internally and would report equilibria computed
without V.  All metrics here go through build_lateral_ops instead.
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from causal_coding.src.activations import relu  # noqa: E402
from causal_coding.src.inference import infer  # noqa: E402
from causal_coding.src.lateral_ops import (  # noqa: E402
    StructLateral,
    build_lateral_ops,
    dense_lateral,
)
from causal_coding.src.method import CausalCodingMethod  # noqa: E402
from causal_coding.src.model import CausalCodingModel  # noqa: E402
from causal_coding.src.vertical_pruning import apply_vertical_gates  # noqa: E402
from core.data import Task  # noqa: E402

ARMS = {
    "L1": dict(lateral_mode="lowrank", struct_learning="fixed"),
    "S0": dict(lateral_mode="structured", struct_learning="fixed"),
    "S1": dict(lateral_mode="structured", struct_learning="scalar"),
    "S2": dict(lateral_mode="structured", struct_learning="perunit"),
}

DIAG_BATCH_SIZE = 256


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_task(smoke, seed):
    if smoke:
        rng = np.random.RandomState(seed)
        n = 512
        X = rng.rand(n, 784).astype(np.float32)
        y = rng.randint(0, 10, size=n)
        return Task(
            train_X=jnp.asarray(X), train_y=jnp.asarray(y),
            test_X=jnp.asarray(X[:128]), test_y=jnp.asarray(y[:128]),
            classes=list(range(10)),
        )
    from core.data import load_mnist  # imports tensorflow lazily

    train_X, train_y, test_X, test_y = load_mnist()
    return Task(
        train_X=jnp.array(train_X), train_y=jnp.array(train_y),
        test_X=jnp.array(test_X), test_y=jnp.array(test_y),
        classes=list(range(10)),
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def forward_eval(model, params, X, y, chunk=5000):
    """Feed-forward accuracy and NLL (what the benchmark evaluates)."""
    correct, nll, n = 0, 0.0, X.shape[0]
    for i in range(0, n, chunk):
        logits = model.forward(params, X[i:i + chunk])
        logp = jax.nn.log_softmax(logits, axis=1)
        yy = y[i:i + chunk]
        correct += int(jnp.sum(jnp.argmax(logits, axis=1) == yy))
        nll += float(-jnp.sum(logp[jnp.arange(yy.shape[0]), yy]))
    return correct / n, nll / n


def equilibrium(model, params, method, X, y, mode, force_scale):
    """Run the training-time relaxation (label clamped) with a given lateral."""
    weights = apply_vertical_gates(
        params["weights"], params["vertical_gate_logits"], params["vertical_layer_scales"]
    )
    precisions = [jnp.exp(lp) for lp in params["log_precisions"]]
    ops = build_lateral_ops(params, mode, force_scale)
    xs = [X.T]
    for l in range(1, model.num_layers - 1):
        xs.append(weights[l - 1] @ relu(xs[l - 1]))
    xs.append(jnp.eye(model.layer_sizes[-1])[y].T)
    return infer(weights, xs, precisions, ops, method.num_inference_steps, method.lr_z)


def phi_within_groups(act, K):
    """Mean phi coefficient between binary activity of units in the same block.

    act: (d, B) float 0/1.  Returns a float (nan-safe).
    """
    d, B = act.shape
    G = d // K
    a = np.asarray(act).reshape(G, K, B)
    p = a.mean(axis=2)                                   # (G, K)
    q = np.einsum("gib,gjb->gij", a, a) / B              # (G, K, K)
    numer = q - p[:, :, None] * p[:, None, :]
    var = p * (1.0 - p)
    denom = np.sqrt(var[:, :, None] * var[:, None, :]) + 1e-12
    r = numer / denom
    off = 1.0 - np.eye(K)[None]
    # Ignore units that are always on / always off (zero variance).
    valid = (var[:, :, None] > 1e-6) & (var[:, None, :] > 1e-6)
    w = off * valid
    if w.sum() == 0:
        return float("nan")
    return float((r * w).sum() / w.sum())


def activity_stats(model, params, method, X, y, mode, K):
    """Active fraction and within-group phi, with the lateral on and off."""
    out = {}
    for tag, scale in (("with_V", 1.0), ("without_V", 0.0)):
        xs = equilibrium(model, params, method, X, y, mode, scale)
        fracs, phis = [], []
        for l in range(1, model.num_layers - 1):
            act = (xs[l] > 0).astype(jnp.float32)
            fracs.append(float(act.mean()))
            phis.append(phi_within_groups(act, K))
        finite = [v for v in xs[1:-1] if not bool(jnp.all(jnp.isfinite(v)))]
        out[f"active_frac_{tag}"] = fracs
        out[f"phi_{tag}"] = phis
        out[f"finite_{tag}"] = len(finite) == 0
        out[f"state_norm_{tag}"] = [float(jnp.linalg.norm(z)) for z in xs[1:-1]]
    return out


def struct_metrics(params):
    rows = []
    for l, (s, b) in enumerate(zip(params["struct_s"], params["struct_b"])):
        G, K, _ = b.shape
        V = dense_lateral(StructLateral(s, b), include_structured=True)
        pi = jnp.exp(params["log_precisions"][l])
        eig_min = float(jnp.linalg.eigvalsh(jnp.diag(pi) + V).min())
        off = 1.0 - jnp.eye(K)[None]
        rows.append({
            "layer": l,
            "s_mean": float(s.mean()),
            "s_max": float(s.max()),
            "b_mean_offdiag": float((b * off).sum() / (off.sum() * G)),
            "b_max": float(b.max()),
            "norm_V_fro": float(jnp.linalg.norm(V)),
            "lambda_min_pi_plus_V": eig_min,
            "p_ema_mean": float(params["struct_p_ema"][l].mean()),
            "guard_frac": float(params["struct_guard_frac"][l]),
        })
    return rows


def lowrank_metrics(params):
    rows = []
    for l, (U, rho) in enumerate(zip(params["lateral_U"], params["lateral_log_alpha"])):
        alpha = float(jax.nn.softplus(rho))
        lam = float(alpha * jnp.linalg.eigvalsh(U.T @ U).max())
        rows.append({"layer": l, "alpha": alpha, "lambda_max": lam,
                     "norm_Lambda_fro": float(alpha * jnp.linalg.norm(U @ U.T))})
    return rows


def snapshot(label, model, params, method, args, mode, task, diag_X, diag_y):
    acc, nll = forward_eval(model, params, task.test_X, task.test_y)
    rec = {"label": label, "test_acc": acc, "test_nll": nll}
    rec.update(activity_stats(model, params, method, diag_X, diag_y, mode, args.group_size))
    if mode in ("structured", "both"):
        rec["struct"] = struct_metrics(params)
    if mode in ("lowrank", "both"):
        rec["lowrank"] = lowrank_metrics(params)
    return rec


def print_record(rec):
    fw = ", ".join(f"{v:.3f}" for v in rec["active_frac_with_V"])
    fo = ", ".join(f"{v:.3f}" for v in rec["active_frac_without_V"])
    pw = ", ".join(f"{v:.3f}" for v in rec["phi_with_V"])
    po = ", ".join(f"{v:.3f}" for v in rec["phi_without_V"])
    print(f"[{rec['label']}] acc={rec['test_acc'] * 100:.2f}%  nll={rec['test_nll']:.4f}")
    print(f"    active frac  with V: [{fw}]   without V: [{fo}]")
    print(f"    phi (groups) with V: [{pw}]   without V: [{po}]")
    if not (rec["finite_with_V"] and rec["finite_without_V"]):
        print("    WARNING: non-finite hidden state in relaxation")
    for r in rec.get("struct", []):
        print(f"    struct L{r['layer']}: s_mean={r['s_mean']:.4f} b_mean={r['b_mean_offdiag']:.4f} "
              f"|V|_F={r['norm_V_fro']:.3f} lam_min(Pi+V)={r['lambda_min_pi_plus_V']:.3f} "
              f"guard_frac={r['guard_frac']:.3f} p_ema={r['p_ema_mean']:.3f}")
    for r in rec.get("lowrank", []):
        print(f"    lowrank L{r['layer']}: alpha={r['alpha']:.2e} lam_max={r['lambda_max']:.2e}")


# --------------------------------------------------------------------------
# Config / output
# --------------------------------------------------------------------------

def vertical_mode_config():
    mode = os.environ.get("VLCP_VERTICAL_MODE", "baseline").strip().lower()
    table = {
        "baseline": (False, (1.0, 1.0, 0.0)),
        "hidden-only": (True, (1.0, 1.0, 0.0)),
        "soft-output": (True, (1.0, 1.0, 0.25)),
        "full": (True, (1.0, 1.0, 1.0)),
        "full-all-layer": (True, (1.0, 1.0, 1.0)),
    }
    if mode not in table:
        raise ValueError(f"Unknown VLCP_VERTICAL_MODE={mode!r}; expected one of {sorted(table)}")
    return (mode,) + table[mode]


def dump_params(params, path):
    out = {}
    for l, w in enumerate(params.get("weights", [])):
        out[f"W_{l}"] = np.asarray(w)
    for l, lp in enumerate(params.get("log_precisions", [])):
        out[f"log_pi_{l}"] = np.asarray(lp)
    for l, U in enumerate(params.get("lateral_U", [])):
        out[f"lateral_U_{l}"] = np.asarray(U)
    for l, rho in enumerate(params.get("lateral_log_alpha", [])):
        out[f"lateral_rho_{l}"] = np.asarray(rho)
    for l, s in enumerate(params.get("struct_s", [])):
        out[f"struct_s_{l}"] = np.asarray(s)
    for l, b in enumerate(params.get("struct_b", [])):
        out[f"struct_b_{l}"] = np.asarray(b)
    for l, p in enumerate(params.get("struct_p_ema", [])):
        out[f"struct_p_ema_{l}"] = np.asarray(p)
    np.savez_compressed(path, **out)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="S0")
    ap.add_argument("--schur-include", action="store_true",
                    help="include the structured matrix in the Schur/gate step (Stage F)")
    ap.add_argument("--rho", type=float, default=0.2,
                    help="target active fraction for the learning rule (placeholder 0.2; "
                         "set from active_frac_without_V of an earlier run)")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--alpha-e", type=float, default=0.13)
    ap.add_argument("--inhib-total", type=float, default=0.25)
    ap.add_argument("--lr-s", type=float, default=1e-3)
    ap.add_argument("--lr-b", type=float, default=1e-3)
    ap.add_argument("--beta-stat", type=float, default=0.99)
    ap.add_argument("--guard-c", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--diag-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny synthetic run (64-unit layers, 2 epochs) to test the script")
    return ap.parse_args()


def main():
    args = parse_args()
    arm = ARMS[args.arm]
    mode = arm["lateral_mode"]
    tag = args.arm + ("+schur" if args.schur_include else "")
    results_dir = args.results_dir or os.path.join(ROOT, "results", "struct_lateral", tag)
    os.makedirs(results_dir, exist_ok=True)

    if args.smoke:
        layer_sizes, args.epochs, args.diag_every = [784, 64, 64, 10], 2, 1
        batch_size = 64
    else:
        layer_sizes, batch_size = [784, 512, 512, 10], 128

    print(f"Arm {tag}: lateral_mode={mode}, struct_learning={arm['struct_learning']}, "
          f"rho={args.rho}, K={args.group_size}")
    task = load_task(args.smoke, args.seed)
    diag_X, diag_y = task.test_X[:DIAG_BATCH_SIZE], task.test_y[:DIAG_BATCH_SIZE]

    model = CausalCodingModel(
        layer_sizes,
        lateral_rank=32 if not args.smoke else 8,
        lateral_init_scale=1e-3,
        alpha_init=0.01,
        lateral_mode=mode,
        struct_group_size=args.group_size,
        struct_alpha_e=args.alpha_e,
        struct_inhib_total=args.inhib_total,
        struct_rho=args.rho,
    )
    params = model.init_params(jax.random.PRNGKey(args.seed))

    vmode, vert_enabled, vert_scales = vertical_mode_config()
    print(f"VLCP vertical mode: {vmode} (enabled={vert_enabled}, layer_scales={vert_scales})")

    # Same settings as run_cc_full_mnist.py, so L1 is directly comparable.
    method_kwargs = dict(
        lr_z=0.05, lr_w=0.02, num_inference_steps=30, gate_p=2.0, gate_kappa=1e-3,
        ridge=1e-4, lambda_s=1e-6, batch_size=batch_size, epochs=args.epochs,
        beta_pi=0.99, k_probe=10,
        lr_lat=1e-3, beta_cov=0.99, eps_lat=1e-2,
        lat_warmup_epochs=5 if not args.smoke else 1,
        lat_ramp_epochs=10 if not args.smoke else 1,
        lambda_max_cap=1.0, beta_logdet=0.1, lambda_fro=1e-2, lambda_U=1e-5,
        lambda_d=0.0, clarity_t=1.0, clarity_eps=1e-4,
        pi0=2.718281828459045, rho_v=0.1, delta_abs=1e-12, d_min=0.75, d_max=1.5,
        vertical_pruning_enabled=vert_enabled, vertical_layer_scales=vert_scales,
        gate_norm="match",
        # structured lateral
        lateral_mode=mode,
        struct_learning=arm["struct_learning"],
        schur_include_structured=args.schur_include,
        struct_lr_s=args.lr_s, struct_lr_b=args.lr_b,
        struct_beta_stat=args.beta_stat, struct_guard_c=args.guard_c,
    )
    if args.smoke:
        method_kwargs.update(k_probe=5, vertical_warmup_epochs=0, vertical_ramp_epochs=1,
                             vertical_importance_update_epochs=1,
                             vertical_importance_batch_size=64,
                             vertical_importance_num_iters=3)
    method = CausalCodingMethod(**method_kwargs)

    records = []
    rec = snapshot("pre", model, params, method, args, mode, task, diag_X, diag_y)
    print_record(rec)
    records.append(rec)

    def epoch_hook(epoch_params, epoch_idx, epoch_loss):
        n = epoch_idx + 1
        if n % args.diag_every != 0 and n != method.epochs:
            return
        r = snapshot(f"epoch_{n}", model, epoch_params, method, args, mode, task, diag_X, diag_y)
        r["loss_clamped"] = float(epoch_loss)
        print_record(r)
        records.append(r)

    print("\nTraining...")
    t0 = time.time()
    params, _, final_loss = method.train_task(
        model, params, None, task, task_idx=0, diagnostics_hook=epoch_hook
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s. Final epoch loss: {final_loss:.4f}")

    post = snapshot("post", model, params, method, args, mode, task, diag_X, diag_y)
    post["loss_clamped"] = float(final_loss)
    post["train_seconds"] = elapsed
    print_record(post)
    records.append(post)

    config = {
        "arm": tag, "args": vars(args), "layer_sizes": layer_sizes,
        "vertical_mode": vmode,
        "method": {k: (list(v) if isinstance(v, tuple) else v) for k, v in method_kwargs.items()},
    }
    out_json = os.path.join(results_dir, "diagnostics.json")
    with open(out_json, "w") as f:
        json.dump({"config": config, "records": records}, f, indent=2)
    dump_params(params, os.path.join(results_dir, "params.npz"))
    print(f"\nWritten: {out_json}")

    # Gate hint for the S arms: how much does V lower the active fraction?
    if mode == "structured":
        w, wo = post["active_frac_with_V"], post["active_frac_without_V"]
        rel = [(b - a) / b if b > 0 else float("nan") for a, b in zip(w, wo)]
        meaningful = all(r >= 0.05 for r in rel)
        print("\nGATE CHECK (S arms): relative drop in active fraction caused by V, per hidden layer: "
              + str([round(r, 3) for r in rel]))
        print("  -> " + ("V lowers activity by >= 5% in every layer: mechanism present."
                         if meaningful else
                         "V lowers activity by < 5% in at least one layer: the effect is weak or absent. "
                         "Consider stopping before S1/S2, or check the scaling of V (--inhib-total, --alpha-e)."))
        print(f"  measured active_frac_without_V (use for --rho): {[round(x, 3) for x in wo]}")


if __name__ == "__main__":
    main()
