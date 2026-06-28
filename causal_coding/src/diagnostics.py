import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu
from causal_coding.src.inference import _infer_step, infer, compute_errors
from causal_coding.src.do_influence import (
    compute_jacobians,
    compute_local_maps,
    compute_causal_gates,
    compute_output_fisher,
)
from causal_coding.src.lateral import (
    diffusion_clarity_kernel,
    materialize_lateral,
)

# Default diffusion-clarity diagnostic parameters.
CLARITY_T_DEFAULT = 1.0
CLARITY_EPS_DEFAULT = 1e-4


def _f(x):
    return float(jnp.asarray(x))


def _initialize_xs(weights, x_batch, y_onehot, num_layers):
    """Initialize states with a feedforward pass and a clamped label state."""
    xs = [x_batch.T]
    for l in range(1, num_layers - 1):
        xs.append(weights[l - 1] @ relu(xs[l - 1]))
    xs.append(y_onehot.T)
    return xs


def _build_lateral_pairs_raw(lateral_U_list, lateral_log_alpha_list):
    """Build full-strength lateral pairs for diagnostics.

    Training may ramp lateral strength over epochs. Diagnostics use the
    underlying parameter value so snapshots are comparable across schedules.
    """
    return [
        (jax.nn.softplus(rho), U)
        for U, rho in zip(lateral_U_list, lateral_log_alpha_list)
    ]


def inference_trajectory(weights, xs_init, precisions, lateral_pairs, num_steps, lr_z):
    """Track per-layer state deltas across the full inference trajectory."""
    num_layers = len(xs_init)
    step_deltas = [[] for _ in range(num_layers)]
    xs = xs_init

    for _t in range(num_steps):
        new_xs = _infer_step(weights, xs, precisions, lateral_pairs, lr_z)
        for l in range(num_layers):
            d = jnp.linalg.norm(new_xs[l] - xs[l])
            step_deltas[l].append(_f(d))
        xs = new_xs

    errors = compute_errors(weights, xs)
    return {
        "step_deltas_per_layer": step_deltas,
        "final_step_delta_per_layer": [d[-1] if d else 0.0 for d in step_deltas],
        "final_state_norms": [_f(jnp.linalg.norm(z)) for z in xs],
        "final_error_norms": [_f(jnp.linalg.norm(e)) for e in errors],
        "relu_active_fraction": [_f(jnp.mean((z > 0).astype(jnp.float32))) for z in xs],
    }


def cc_machinery_stats(
    weights, xs_eq, precisions, lateral_pairs,
    gate_p, gate_kappa, ridge, gate_alpha, gate_floor_coeff,
    output_fisher=None,
):
    """Per-layer Jacobian / A_tilde norms, condition numbers, gate stats.

    Returns the stats dict and the raw (jacobians, A_tildes, gates) so the
    caller can reuse them without recomputing.
    """
    jacobians = compute_jacobians(weights, xs_eq)
    A_tildes = compute_local_maps(
        precisions, lateral_pairs, jacobians, ridge, output_fisher=output_fisher
    )
    gates = compute_causal_gates(
        A_tildes, gate_p, gate_kappa, gate_alpha, gate_floor_coeff
    )

    layers = []
    for l, (J, A, G) in enumerate(zip(jacobians, A_tildes, gates)):
        n_in = G.shape[1]
        eps = 1e-12
        row_sum = jnp.sum(G, axis=1, keepdims=True) + eps
        G_dist = G / row_sum
        row_entropy = -jnp.sum(G_dist * jnp.log(G_dist + eps), axis=1)
        mean_g = jnp.mean(G)
        try:
            s = jnp.linalg.svd(A, compute_uv=False)
            cond = float(s[0] / (s[-1] + 1e-12))
        except Exception:
            cond = float("nan")
        layers.append({
            "J_fro": _f(jnp.linalg.norm(J)),
            "A_tilde_fro": _f(jnp.linalg.norm(A)),
            "A_tilde_cond": cond,
            "gate_mean": _f(mean_g),
            "gate_std": _f(jnp.std(G)),
            "gate_max": _f(jnp.max(G)),
            "gate_min": _f(jnp.min(G)),
            "frac_above_2x_mean": _f(jnp.mean((G > 2.0 * mean_g).astype(jnp.float32))),
            "frac_above_5x_mean": _f(jnp.mean((G > 5.0 * mean_g).astype(jnp.float32))),
            "row_entropy_mean": _f(jnp.mean(row_entropy)),
            "row_entropy_max_possible": float(jnp.log(n_in)),
            "row_entropy_min": _f(jnp.min(row_entropy)),
            "row_entropy_max": _f(jnp.max(row_entropy)),
        })

    return {"layers": layers}, jacobians, A_tildes, gates


def output_fisher_stats(F_out, ridge):
    """Stats on the categorical-Fisher output curvature."""
    eps = 1e-12
    d = F_out.shape[0]
    eig = jnp.linalg.eigvalsh(F_out)
    eig_min_ridged = float(jnp.min(eig) + ridge)
    eig_max_ridged = float(jnp.max(eig) + ridge)
    cond = eig_max_ridged / (eig_min_ridged + eps)
    return {
        "trace": _f(jnp.trace(F_out)),
        "eig_min": _f(jnp.min(eig)),
        "eig_max": _f(jnp.max(eig)),
        "eig_min_ridged": eig_min_ridged,
        "eig_max_ridged": eig_max_ridged,
        "condition_number_ridged": cond,
        "rank_estimate": int(jnp.sum((eig > 1e-6).astype(jnp.int32))),
        "dim": int(d),
    }


def weight_update_stats(
    weights, xs_eq, errors, gates, precisions, batch_size, dead_threshold=1e-4
):
    """Hebbian update magnitudes, gate-induced shrinkage, weight norms."""
    layers = []
    for l in range(len(weights)):
        weighted_error = precisions[l][:, None] * errors[l + 1]
        delta_w = (1.0 / batch_size) * (weighted_error @ relu(xs_eq[l]).T)
        gated_delta_w = gates[l] * delta_w
        delta_norm = jnp.linalg.norm(delta_w)
        gated_norm = jnp.linalg.norm(gated_delta_w)
        layers.append({
            "delta_w_fro": _f(delta_norm),
            "gated_delta_w_fro": _f(gated_norm),
            "shrinkage_ratio": _f(gated_norm / (delta_norm + 1e-12)),
            "W_fro": _f(jnp.linalg.norm(weights[l])),
            "W_dead_frac": _f(
                jnp.mean((jnp.abs(weights[l]) < dead_threshold).astype(jnp.float32))
            ),
        })
    return {"layers": layers}


def precision_stats(log_precisions, errors, precision_var_ema=None):
    """log-precision distribution + residual MSE + clip-saturation fraction."""
    layers = []
    num_hidden_ema = 0 if precision_var_ema is None else len(precision_var_ema)
    for l, lp in enumerate(log_precisions):
        pi = jnp.exp(lp)
        mse = jnp.mean(errors[l + 1] ** 2, axis=1)
        entry = {
            "log_pi_mean": _f(jnp.mean(lp)),
            "log_pi_min": _f(jnp.min(lp)),
            "log_pi_max": _f(jnp.max(lp)),
            "log_pi_std": _f(jnp.std(lp)),
            "mse_mean": _f(jnp.mean(mse)),
            "pi_mse_mean": _f(jnp.mean(pi * mse)),
            "frac_at_neg_clip": _f(jnp.mean((lp <= -3.99).astype(jnp.float32))),
            "frac_at_pos_clip": _f(jnp.mean((lp >= 3.99).astype(jnp.float32))),
        }
        if l < num_hidden_ema:
            v = precision_var_ema[l]
            entry["var_ema_mean"] = _f(jnp.mean(v))
            entry["var_ema_min"] = _f(jnp.min(v))
            entry["var_ema_max"] = _f(jnp.max(v))
            entry["var_ema_std"] = _f(jnp.std(v))
        layers.append(entry)
    return {"layers": layers}


def _batched_forward_logits(model, params, X, batch_size=1024):
    """Forward pass in chunks to avoid OOM on the full train set."""
    N = X.shape[0]
    chunks = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunks.append(model.forward(params, X[start:end]))
    return jnp.concatenate(chunks, axis=0)


def forward_accuracy(model, params, X, y, batch_size=1024):
    """Feedforward accuracy + NLL on the given dataset, batched."""
    logits = _batched_forward_logits(model, params, X, batch_size)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    preds = jnp.argmax(logits, axis=1)
    acc = float(jnp.mean(preds == y))
    nll = float(-jnp.mean(log_probs[jnp.arange(y.shape[0]), y]))
    return acc, nll, preds


def per_class_accuracy(preds, y, num_classes):
    """Per-class accuracy on the given predictions (returns a length-K list)."""
    accs = []
    y_int = y.astype(jnp.int32)
    preds_int = preds.astype(jnp.int32)
    for c in range(num_classes):
        mask = (y_int == c).astype(jnp.float32)
        total = jnp.sum(mask)
        correct = jnp.sum(((preds_int == c).astype(jnp.float32)) * mask)
        accs.append(float(correct / (total + 1e-12)))
    return accs


def equilibrium_vs_feedforward_gap(model, params, method, X, y):
    """Per-layer divergence between equilibrium (label-clamped) and feedforward.

    The output layer will diverge because equilibrium clamps the label while
    feedforward evaluation leaves the output as logits. Hidden layers give
    the useful comparison.
    """
    num_classes = model.layer_sizes[-1]
    y_onehot = jnp.eye(num_classes)[y]

    weights = params["weights"]
    log_precisions = params["log_precisions"]
    num_layers = len(weights) + 1

    precisions = [jnp.exp(lp) for lp in log_precisions]
    lateral_pairs = _build_lateral_pairs_raw(
        params["lateral_U"], params["lateral_log_alpha"]
    )

    # Feedforward states before label-clamped inference.
    xs_ff = [X.T]
    for l in range(1, num_layers):
        xs_ff.append(weights[l - 1] @ relu(xs_ff[l - 1]))

    xs_init = _initialize_xs(weights, X, y_onehot, num_layers)
    xs_eq = infer(
        weights, xs_init, precisions, lateral_pairs,
        method.num_inference_steps, method.lr_z,
    )

    layers = []
    for l in range(num_layers):
        ff = xs_ff[l]
        eq = xs_eq[l]
        ff_flat = ff.reshape(-1)
        eq_flat = eq.reshape(-1)
        ff_norm_v = jnp.linalg.norm(ff_flat) + 1e-12
        eq_norm_v = jnp.linalg.norm(eq_flat) + 1e-12
        diff_norm = jnp.linalg.norm(eq - ff)
        cos_sim = jnp.sum(ff_flat * eq_flat) / (ff_norm_v * eq_norm_v)
        layers.append({
            "ff_norm": _f(jnp.linalg.norm(ff)),
            "eq_norm": _f(jnp.linalg.norm(eq)),
            "diff_norm": _f(diff_norm),
            "rel_diff": _f(diff_norm / ff_norm_v),
            "cosine_sim": _f(cos_sim),
        })
    return {"layers": layers}


def lateral_clarity_stats(
    lateral_U_list, lateral_log_alpha_list,
    t=CLARITY_T_DEFAULT, eps=CLARITY_EPS_DEFAULT,
):
    """
    heat-kernel diagnostic (VLCP).

    For each hidden layer with Λ_l = softplus(ρ_l) · U_l U_lᵀ, build
    the unnormalised graph Laplacian L = D − |Λ| (diagonal of |Λ|
    included), form the heat kernel K_t = exp(−t · L), and
    report scalar summaries of the candidate penalty integrand
    `(K_t − |Λ| − ε)_+` restricted to off-diagonal entries (u ≠ v).

    Pure diagnostic — does not enter the training path.
    """
    layers = []
    for U, rho in zip(lateral_U_list, lateral_log_alpha_list):
        raw_alpha = jax.nn.softplus(rho)
        K_t = diffusion_clarity_kernel(U, raw_alpha, t)
        abs_Lam = jnp.abs(materialize_lateral(U, raw_alpha))
        raw_gap = K_t - abs_Lam

        d = U.shape[0]
        offdiag_mask = 1.0 - jnp.eye(d)
        n_offdiag = float(d * (d - 1))

        K_t_fro = jnp.linalg.norm(K_t)
        K_t_offdiag_sum_abs = jnp.sum(jnp.abs(K_t) * offdiag_mask)
        K_t_offdiag_mean_abs = K_t_offdiag_sum_abs / n_offdiag

        # Signed off-diagonal gap statistics.
        raw_gap_offdiag = raw_gap * offdiag_mask
        raw_gap_offdiag_mean = jnp.sum(raw_gap_offdiag) / n_offdiag
        # Mask diagonal to a very negative value before taking the max so
        # off-diagonal entries always win the argmax.
        raw_gap_max = jnp.max(raw_gap - 1e12 * jnp.eye(d))

        # Penalty integrand: (K_t − |Λ| − ε)_+ restricted to u ≠ v.
        penalty_pos = jax.nn.relu(raw_gap - eps) * offdiag_mask
        penalty_value = jnp.sum(penalty_pos)
        active_mask = (penalty_pos > 0.0).astype(jnp.float32)
        penalty_active_count = jnp.sum(active_mask)
        penalty_active_frac = penalty_active_count / n_offdiag
        penalty_mean_active = jnp.where(
            penalty_active_count > 0,
            penalty_value / jnp.maximum(penalty_active_count, 1.0),
            jnp.asarray(0.0),
        )

        layers.append({
            "t": float(t),
            "eps": float(eps),
            "K_t_fro": _f(K_t_fro),
            "K_t_offdiag_mean_abs": _f(K_t_offdiag_mean_abs),
            "raw_gap_offdiag_mean": _f(raw_gap_offdiag_mean),
            "raw_gap_offdiag_max": _f(raw_gap_max),
            "penalty_value": _f(penalty_value),
            "penalty_active_count": int(_f(penalty_active_count)),
            "penalty_active_frac": _f(penalty_active_frac),
            "penalty_mean_active": _f(penalty_mean_active),
        })

    return {"layers": layers, "t": float(t), "eps": float(eps)}


def lateral_stats(
    weights, xs_eq, errors, precisions,
    lateral_U_list, lateral_log_alpha_list, lateral_cov_ema_list,
    jacobians,
):
    """Per-hidden-layer low-rank lateral diagnostics."""
    layers = []
    num_hidden = len(lateral_U_list)

    for l_idx in range(num_hidden):
        U = lateral_U_list[l_idx]
        rho = lateral_log_alpha_list[l_idx]
        C_ema = lateral_cov_ema_list[l_idx]

        raw_alpha = jax.nn.softplus(rho)
        UtU = U.T @ U
        eigs = jnp.linalg.eigvalsh(UtU)
        lam_max_UtU = jnp.max(eigs)
        lam_max_Lambda = raw_alpha * lam_max_UtU

        # ||Λ||_F = α · ||Uᵀ U||_F
        UtU_fro = jnp.linalg.norm(UtU)
        Lambda_fro = raw_alpha * UtU_fro
        U_fro = jnp.linalg.norm(U)

        # Lateral force at equilibrium: Λ z = α · U (Uᵀ z), shape (d, batch).
        z_l = xs_eq[l_idx + 1]
        lateral_force_vec = raw_alpha * (U @ (U.T @ z_l))
        lateral_force_norm = jnp.linalg.norm(lateral_force_vec)

        # Vertical predictive-coding force at equilibrium.
        pi_self = precisions[l_idx]
        eps_self = errors[l_idx + 1]
        weighted_self = pi_self[:, None] * eps_self
        self_force_norm = jnp.linalg.norm(weighted_self)

        pi_above = precisions[l_idx + 1]
        eps_above = errors[l_idx + 2]

        # Top-down passes through Jᵀ at the layer above.
        weighted_above = pi_above[:, None] * eps_above
        top_down_force = jacobians[l_idx + 1].T @ weighted_above
        top_down_force_norm = jnp.linalg.norm(top_down_force)

        vertical_force_norm = self_force_norm + top_down_force_norm
        R_lat = lateral_force_norm / (vertical_force_norm + 1e-12)

        # Covariance EMA stats.
        d = C_ema.shape[0]
        cov_diag = jnp.diag(C_ema)
        cov_offdiag_mask = 1.0 - jnp.eye(d)
        cov_offdiag = C_ema * cov_offdiag_mask
        cov_offdiag_mean_abs = jnp.sum(jnp.abs(cov_offdiag)) / (d * (d - 1) + 1e-12)

        layers.append({
            "alpha_raw": _f(raw_alpha),
            "rho": _f(rho),
            "U_fro": _f(U_fro),
            "Lambda_fro": _f(Lambda_fro),
            "Lambda_lambda_max": _f(lam_max_Lambda),
            "UtU_lambda_max": _f(lam_max_UtU),
            "Lambda_z_norm": _f(lateral_force_norm),
            "vertical_self_force_norm": _f(self_force_norm),
            "vertical_topdown_force_norm": _f(top_down_force_norm),
            "vertical_force_norm": _f(vertical_force_norm),
            "force_ratio_R_lat": _f(R_lat),
            "cov_ema_trace": _f(jnp.trace(C_ema)),
            "cov_ema_diag_mean": _f(jnp.mean(cov_diag)),
            "cov_ema_diag_max": _f(jnp.max(cov_diag)),
            "cov_ema_offdiag_mean_abs": _f(cov_offdiag_mean_abs),
        })
    return {"layers": layers}


def run_full_diagnostics(
    model, params, method,
    diag_X, diag_y,
    train_X, train_y,
    test_X, test_y,
    label,
    include_structural=False,
):
    """Run all diagnostic levels and return a nested dict."""
    weights = params["weights"]
    log_precisions = params["log_precisions"]
    precision_var_ema = params.get("precision_var_ema")
    lateral_U_list = params.get("lateral_U", [])
    lateral_log_alpha_list = params.get("lateral_log_alpha", [])
    lateral_cov_ema_list = params.get("lateral_cov_ema", [])
    num_layers = len(weights) + 1

    precisions = [jnp.exp(lp) for lp in log_precisions]
    lateral_pairs = _build_lateral_pairs_raw(lateral_U_list, lateral_log_alpha_list)

    num_classes = model.layer_sizes[-1]
    y_onehot = jnp.eye(num_classes)[diag_y]

    xs_init = _initialize_xs(weights, diag_X, y_onehot, num_layers)

    # Inference convergence with full-strength lateral parameters.
    inf_diag = inference_trajectory(
        weights, xs_init, precisions, lateral_pairs,
        method.num_inference_steps, method.lr_z,
    )

    # Equilibrium state used by the remaining diagnostics.
    xs_eq = infer(
        weights, xs_init, precisions, lateral_pairs,
        method.num_inference_steps, method.lr_z,
    )
    errors = compute_errors(weights, xs_eq)

    # Causal influence maps and gates.
    output_fisher = compute_output_fisher(weights, xs_eq)
    current_gate_alpha = getattr(method, "gate_alpha_current", method.gate_alpha)
    cc_diag, jacobians, _A_tildes, gates = cc_machinery_stats(
        weights, xs_eq, precisions, lateral_pairs,
        method.gate_p, method.gate_kappa, method.ridge,
        current_gate_alpha, method.gate_floor_coeff,
        output_fisher=output_fisher,
    )
    of_diag = output_fisher_stats(output_fisher, method.ridge)

    # Weight-update magnitudes.
    wu_diag = weight_update_stats(
        weights, xs_eq, errors, gates, precisions, diag_X.shape[0]
    )

    # Precision and residual statistics.
    pi_diag = precision_stats(log_precisions, errors, precision_var_ema)

    # Feedforward classification metrics.
    train_acc, train_nll, _ = forward_accuracy(model, params, train_X, train_y)
    test_acc, test_nll, test_preds = forward_accuracy(model, params, test_X, test_y)
    pc_acc = per_class_accuracy(test_preds, test_y, num_classes)
    perf_diag = {
        "train_acc": train_acc,
        "train_nll": train_nll,
        "test_acc": test_acc,
        "test_nll": test_nll,
        "per_class_test_acc": pc_acc,
    }

    # Lateral precision statistics.
    lat_diag = lateral_stats(
        weights, xs_eq, errors, precisions,
        lateral_U_list, lateral_log_alpha_list, lateral_cov_ema_list,
        jacobians,
    )

    # Diffusion-clarity diagnostic 
    clarity_t_used = float(getattr(method, "clarity_t", CLARITY_T_DEFAULT))
    clarity_eps_used = float(getattr(method, "clarity_eps", CLARITY_EPS_DEFAULT))
    if lateral_U_list:
        clarity_diag = lateral_clarity_stats(
            lateral_U_list, lateral_log_alpha_list,
            t=clarity_t_used, eps=clarity_eps_used,
        )
    else:
        clarity_diag = {"layers": [], "t": clarity_t_used, "eps": clarity_eps_used}

    out = {
        "label": label,
        "inference": inf_diag,
        "cc_machinery": cc_diag,
        "output_fisher": of_diag,
        "weight_update": wu_diag,
        "precision": pi_diag,
        "lateral": lat_diag,
        "lateral_clarity": clarity_diag,
        "performance": perf_diag,
    }

    # Optional comparison between feedforward states and inferred states.
    if include_structural:
        out["structural"] = equilibrium_vs_feedforward_gap(
            model, params, method, diag_X, diag_y
        )

    return out


def pprint_diag_summary(diag, header=None):
    """Compact, human-readable summary of one diagnostic snapshot."""
    if header is not None:
        print(header)
    p = diag["performance"]
    inf = diag["inference"]
    cc = diag["cc_machinery"]["layers"]
    wu = diag["weight_update"]["layers"]
    pi = diag["precision"]["layers"]
    lat = diag.get("lateral", {}).get("layers", [])

    print(
        f"  Forward acc:   train={p['train_acc'] * 100:6.2f}%   "
        f"test={p['test_acc'] * 100:6.2f}%   "
        f"NLL_test={p['test_nll']:.4f}"
    )

    final_deltas = inf["final_step_delta_per_layer"]
    relu_active = inf["relu_active_fraction"]
    deltas_str = " ".join(f"{d:.2e}" for d in final_deltas)
    active_str = " ".join(f"{a:.2f}" for a in relu_active)
    print(f"  Inference:     final_delta=[{deltas_str}]   relu_active=[{active_str}]")

    gate_str = "   ".join(
        f"L{l} frac>2x_mean={c['frac_above_2x_mean']:.3f}"
        for l, c in enumerate(cc)
    )
    print(f"  Gates:         {gate_str}")

    ent_str = "   ".join(
        f"L{l} row_H={c['row_entropy_mean']:.2f}/{c['row_entropy_max_possible']:.2f}"
        for l, c in enumerate(cc)
    )
    print(f"  Gate entropy:  {ent_str}")

    shrink_str = "   ".join(
        f"L{l}={w['shrinkage_ratio']:.3f}"
        for l, w in enumerate(wu)
    )
    print(f"  Update shrink: {shrink_str}")

    pi_parts = []
    for l, x in enumerate(pi):
        chunk = f"L{l} log_pi={x['log_pi_mean']:.2f} pi*mse={x['pi_mse_mean']:.3f}"
        if "var_ema_mean" in x:
            chunk += f" var={x['var_ema_mean']:.2e}"
        pi_parts.append(chunk)
    print(f"  Precision:     {'   '.join(pi_parts)}")

    of = diag.get("output_fisher")
    if of is not None:
        print(
            f"  Output Fisher: tr={of['trace']:.4f}  "
            f"eig[min,max]=[{of['eig_min']:.2e}, {of['eig_max']:.2e}]  "
            f"cond(F+ridge)={of['condition_number_ridged']:.2e}  "
            f"rank={of['rank_estimate']}/{of['dim']}"
        )

    if lat:
        lat_parts = []
        for l, x in enumerate(lat):
            lat_parts.append(
                f"L{l} α={x['alpha_raw']:.2e} ||Λ||={x['Lambda_fro']:.2e} "
                f"λmax={x['Lambda_lambda_max']:.2e} R={x['force_ratio_R_lat']:.2e}"
            )
        print(f"  Lateral:       {'   '.join(lat_parts)}")

        cov_parts = []
        for l, x in enumerate(lat):
            cov_parts.append(
                f"L{l} tr={x['cov_ema_trace']:.2e} "
                f"diag={x['cov_ema_diag_mean']:.2e} "
                f"|off|={x['cov_ema_offdiag_mean_abs']:.2e}"
            )
        print(f"  Cov EMA:       {'   '.join(cov_parts)}")

    clarity = diag.get("lateral_clarity", {}).get("layers", [])
    if clarity:
        clarity_parts = []
        for l, x in enumerate(clarity):
            clarity_parts.append(
                f"L{l} active={x['penalty_active_count']} "
                f"({x['penalty_active_frac'] * 100:.1f}%) "
                f"pen={x['penalty_value']:.2e}"
            )
        print(f"  Lat clarity:   {'   '.join(clarity_parts)}")

    pc = p["per_class_test_acc"]
    pc_str = " ".join(f"{a * 100:5.1f}" for a in pc)
    print(f"  Per-class:     [{pc_str}]")

    if "structural" in diag:
        s = diag["structural"]["layers"]
        # Hidden layers only (skip input l=0 and output l=L).
        rel_diffs = "   ".join(
            f"L{l} rel={x['rel_diff']:.2f} cos={x['cosine_sim']:.2f}"
            for l, x in enumerate(s)
            if 0 < l < len(s) - 1
        )
        print(f"  Equilib-vs-FF: {rel_diffs}")
