import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu
from causal_coding.src.inference import (
    _infer_step,
    infer,
    infer_with_trajectory,
    compute_errors,
)
from causal_coding.src.do_influence import (
    compute_jacobians,
    compute_jacobians_per_sample,
    compute_local_maps,
    compute_causal_gates,
    compute_causal_gates_per_sample,
    compute_output_fisher,
)
from causal_coding.src.lateral import (
    diffusion_clarity_kernel,
    materialize_lateral,
)
from causal_coding.src.vlcp_regression import compute_dreg_for_pair
from causal_coding.src.vertical_pruning import (
    apply_vertical_gates,
    vertical_pruning_stats,
)

# Default diffusion-clarity diagnostic parameters.
CLARITY_T_DEFAULT = 1.0
CLARITY_EPS_DEFAULT = 1e-4

# VLCP regression diagnostic defaults.
VLCP_LAMBDA_DYN_DEFAULT = 1e-3
VLCP_WINDOW_DEFAULT = 15
VLCP_NUM_ITERS_DEFAULT = 50
VLCP_DREG_ZERO_THRESHOLD = 1e-6
VLCP_WEIGHT_DEAD_THRESHOLD = 1e-4
VLCP_SWEEP_LAMBDAS_DEFAULT = (1e-5, 1e-4, 1e-3)
VLCP_SWEEP_WINDOWS_DEFAULT = ((0, 15), (5, 20), (15, 30), (0, 30))
VLCP_SWEEP_STANDARDIZE_X_DEFAULT = (False, True)


def _f(x):
    return float(jnp.asarray(x))


def _effective_weights_from_params(params):
    weights = params["weights"]
    if "vertical_gate_logits" not in params or "vertical_layer_scales" not in params:
        return weights
    return apply_vertical_gates(
        weights,
        params["vertical_gate_logits"],
        params["vertical_layer_scales"],
    )


def _initialize_xs(weights, x_batch, y_onehot, num_layers):
    """Initialize states with a feedforward pass and a clamped label state."""
    xs = [x_batch.T]
    for l in range(1, num_layers - 1):
        xs.append(weights[l - 1] @ relu(xs[l - 1]))
    xs.append(y_onehot.T)
    return xs


def _relative_hidden_deltas(xs_old, xs_new, eps=1e-8):
    """Relative per-hidden-layer state update magnitudes."""
    deltas = []
    for l in range(1, len(xs_old) - 1):
        denom = jnp.linalg.norm(xs_old[l]) + eps
        deltas.append(jnp.linalg.norm(xs_new[l] - xs_old[l]) / denom)
    return deltas


def _build_lateral_pairs_raw(lateral_U_list, lateral_log_alpha_list):
    """Build full-strength lateral pairs for diagnostics.

    Training may ramp lateral strength over epochs. Diagnostics use the
    underlying parameter value so snapshots are comparable across schedules.
    """
    return [
        (jax.nn.softplus(rho), U)
        for U, rho in zip(lateral_U_list, lateral_log_alpha_list)
    ]


def inference_trajectory(
    weights,
    xs_init,
    precisions,
    lateral_pairs,
    num_steps,
    lr_z,
):
    """Track per-layer state deltas and return `(diagnostics, final_state)`."""
    num_layers = len(xs_init)
    step_deltas = [[] for _ in range(num_layers)]
    relative_step_deltas = [[] for _ in range(num_layers)]
    xs = xs_init
    final_relative_delta = float("inf")

    for t in range(num_steps):
        new_xs = _infer_step(weights, xs, precisions, lateral_pairs, lr_z)
        rels_hidden = _relative_hidden_deltas(xs, new_xs)
        for l in range(num_layers):
            d = jnp.linalg.norm(new_xs[l] - xs[l])
            step_deltas[l].append(_f(d))
            if 0 < l < num_layers - 1:
                rel = rels_hidden[l - 1]
            else:
                rel = jnp.asarray(0.0)
            relative_step_deltas[l].append(_f(rel))
        final_relative_delta = (
            max(relative_step_deltas[l][-1] for l in range(1, num_layers - 1))
            if num_layers > 2 else 0.0
        )
        xs = new_xs

    errors = compute_errors(weights, xs)
    diag = {
        "step_deltas_per_layer": step_deltas,
        "final_step_delta_per_layer": [d[-1] if d else 0.0 for d in step_deltas],
        "relative_step_deltas_per_layer": relative_step_deltas,
        "final_relative_delta_per_layer": [
            d[-1] if d else 0.0 for d in relative_step_deltas
        ],
        "final_state_norms": [_f(jnp.linalg.norm(z)) for z in xs],
        "final_error_norms": [_f(jnp.linalg.norm(e)) for e in errors],
        "relu_active_fraction": [_f(jnp.mean((z > 0).astype(jnp.float32))) for z in xs],
        "steps_used": int(num_steps),
        "final_relative_delta": float(final_relative_delta),
    }
    return diag, xs


def cc_machinery_stats(
    weights, xs_eq, precisions, lateral_pairs,
    gate_p, gate_kappa, ridge,
    output_fisher=None,
):
    """Per-layer Jacobian / A_tilde norms, condition numbers, gate stats.

    Returns the stats dict and the raw (jacobians, A_tildes, gates) so the
    caller can reuse them without recomputing.

    Ã is computed per-sample from per-sample Jacobians
    (`compute_jacobians_per_sample`); the gate then batch-averages `|Ã|^p`.
    Extra per-layer fields report:
      - `per_sample_A_std_over_mean`: how much per-sample Ã differs from
        its batch mean (near 0 ⇒ no-op, > 0.3 ⇒ per-sample structure is used).
      - `A_batch_averaged_vs_persample_diff`: |mean_b Ã^{(b)} − Ã_batch_avg|
        relative to |Ã_batch_avg|, where `Ã_batch_avg` is the
        batch-averaged-J baseline. Sanity check.
    """
    jacobians = compute_jacobians(weights, xs_eq)
    jacobians_per_sample = compute_jacobians_per_sample(weights, xs_eq)
    A_tildes_per_sample = compute_local_maps(
        precisions, lateral_pairs, jacobians, ridge, output_fisher=output_fisher,
        jacobians_per_sample=jacobians_per_sample,
    )
    A_tildes_batch = compute_local_maps(
        precisions, lateral_pairs, jacobians, ridge, output_fisher=output_fisher,
    )
    # Training-side gates are per-sample; diagnostics keep a 2D summary
    # from mean-|Ã|^p for row_H / gate_max / row_sum continuity.
    gates_per_sample = compute_causal_gates_per_sample(
        A_tildes_per_sample, gate_p, gate_kappa
    )
    gates = compute_causal_gates(A_tildes_per_sample, gate_p, gate_kappa)

    # Represent Ã to downstream callers (VLCP regression, etc.) as the
    # batch-mean of the per-sample Ã. Callers expect a list of 2D
    # `(d_out, d_in)` arrays; the batch-mean is the natural summary and it
    # is also what the regression diagnostic averages over trajectories to test
    # against.
    A_tildes = [jnp.mean(A, axis=0) for A in A_tildes_per_sample]

    layers = []
    for l, (J, A_per, A_batch, A_mean_l, G, G_per) in enumerate(
        zip(jacobians, A_tildes_per_sample, A_tildes_batch, A_tildes, gates,
            gates_per_sample)
    ):
        n_in = G.shape[1]
        eps = 1e-12
        row_sum = jnp.sum(G, axis=1)                    # raw row sum ≈ 1 − κ/denom
        row_sum_safe = row_sum[:, None] + eps
        G_dist = G / row_sum_safe
        row_entropy = -jnp.sum(G_dist * jnp.log(G_dist + eps), axis=1)
        mean_g = jnp.mean(G)

        try:
            s = jnp.linalg.svd(A_mean_l, compute_uv=False)
            cond = float(s[0] / (s[-1] + 1e-12))
        except Exception:
            cond = float("nan")

        # Per-sample engagement diagnostic: how much does Ã vary across samples?
        A_center = A_per - A_mean_l[None, :, :]                              # (B, d_out, d_in)
        per_sample_fro = jnp.sqrt(jnp.mean(A_center ** 2, axis=(1, 2)))       # (B,)
        A_per_fro = jnp.sqrt(jnp.mean(A_per ** 2, axis=(1, 2)))                # (B,)
        per_sample_std_over_mean = jnp.mean(per_sample_fro) / (
            jnp.mean(A_per_fro) + eps
        )

        # Sanity: how far is mean_b(Ã^{(b)}) from the batch-averaged-J Ã?
        A_batch_avg_norm = jnp.linalg.norm(A_batch) + eps
        A_diff_norm = jnp.linalg.norm(A_mean_l - A_batch)
        batch_vs_persample_rel = A_diff_norm / A_batch_avg_norm

        # Per-sample gate diagnostics.
        # G_per shape: (B, d_out, d_in). Per-sample row-sum: (B, d_out).
        G_per_row_sum = jnp.sum(G_per, axis=2)                                  # (B, d_out)
        G_per_row_sum_safe = G_per_row_sum[..., None] + eps                     # (B, d_out, 1)
        G_per_dist = G_per / G_per_row_sum_safe                                 # (B, d_out, d_in)
        G_per_row_entropy = -jnp.sum(
            G_per_dist * jnp.log(G_per_dist + eps), axis=2
        )                                                                        # (B, d_out)
        per_sample_row_H_mean = jnp.mean(G_per_row_entropy)
        per_sample_gate_max_mean = jnp.mean(jnp.max(G_per, axis=(1, 2)))

        # Agreement between the training-time per-sample gate and the 2D
        # summary. Small means the per-sample form collapses to the summary;
        # large means the per-sample form is genuinely different.
        G_per_mean_over_batch = jnp.mean(G_per, axis=0)                          # (d_out, d_in)
        G_summary_norm = jnp.linalg.norm(G) + eps
        per_sample_gate_agreement = (
            jnp.linalg.norm(G_per_mean_over_batch - G) / G_summary_norm
        )

        layers.append({
            "J_fro": _f(jnp.linalg.norm(J)),
            "A_tilde_fro": _f(jnp.linalg.norm(A_mean_l)),
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
            # Row-sum for normalized causal gates: ≈ 1 − κ/(Σ|Ã|^p + κ).
            "row_sum_mean": _f(jnp.mean(row_sum)),
            "row_sum_min": _f(jnp.min(row_sum)),
            "row_sum_max": _f(jnp.max(row_sum)),
            # Per-sample Ã diagnostics.
            "per_sample_A_std_over_mean": _f(per_sample_std_over_mean),
            "A_batch_averaged_vs_persample_diff": _f(batch_vs_persample_rel),
            # Per-sample gate diagnostics.
            "per_sample_gate_max_mean": _f(per_sample_gate_max_mean),
            "per_sample_row_H_mean": _f(per_sample_row_H_mean),
            "per_sample_gate_agreement": _f(per_sample_gate_agreement),
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
    weights,
    xs_eq,
    errors,
    gates,
    precisions,
    batch_size,
    dead_threshold=1e-4,
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


def precision_stats(
    log_precisions, errors, precision_var_ema=None,
    d_min=None, d_max=None,
):
    """log-precision distribution + residual MSE + clip-saturation fraction.

    When `d_min` / `d_max` are provided, also reports per-layer D statistics
    (the mean-1 relative diagonal used by structured precision). D is recovered
    from log_pi via `D = exp(log_pi) / mean(exp(log_pi))`, so the D block is
    well-defined regardless of which precision-update path produced log_pi
    — under the uniform-π path, D collapses to ones.
    """
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

        # D statistics — only meaningful for hidden layers (output stays
        # frozen at log_pi = 0). For the output layer the block reports D
        # against the single-entry mean, which by construction gives D ≡ 1.
        d_recovered = pi / (jnp.mean(pi) + 1e-12)
        entry["D_mean"] = _f(jnp.mean(d_recovered))
        entry["D_min"] = _f(jnp.min(d_recovered))
        entry["D_max"] = _f(jnp.max(d_recovered))
        entry["D_std"] = _f(jnp.std(d_recovered))
        if d_min is not None and d_max is not None:
            # Saturation proxy: fraction of units within 5 % of either bound.
            # Tight-bound proximity flags clip pressure even when no entry
            # is exactly pinned.
            low_band = jnp.asarray(d_min) * 1.05
            high_band = jnp.asarray(d_max) * 0.95
            entry["frac_at_d_min"] = _f(
                jnp.mean((d_recovered <= low_band).astype(jnp.float32))
            )
            entry["frac_at_d_max"] = _f(
                jnp.mean((d_recovered >= high_band).astype(jnp.float32))
            )
            entry["d_min_cfg"] = float(d_min)
            entry["d_max_cfg"] = float(d_max)
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

    weights = _effective_weights_from_params(params)
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
    VLCP heat-kernel diagnostic.

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


def _pearson_corr_flat(a, b):
    """Pearson correlation for same-shaped arrays, robust to zero variance."""
    a = jnp.ravel(a)
    b = jnp.ravel(b)
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    denom = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    return jnp.where(
        denom > 1e-12,
        jnp.sum(a * b) / (denom + 1e-12),
        jnp.asarray(float("nan")),
    )


def _output_logit_trajectory(weights, trajectory):
    """Logit trajectory for the clamped output edge diagnostic.

    The output state xs[-1] is label-clamped during inference, so literal
    Delta z_output is identically zero. For the output layer pair only, this
    records Delta logits while keeping z_{L-1}(t) as the predictor.
    """
    hidden_traj = relu(trajectory[-2])
    return jnp.einsum("od,dbt->obt", weights[-1], hidden_traj)


def _safe_masked_mean(values, mask):
    mask_f = mask.astype(jnp.float32)
    denom = jnp.sum(mask_f)
    return jnp.where(
        denom > 0.0,
        jnp.sum(values * mask_f) / (denom + 1e-12),
        jnp.asarray(0.0),
    )


def _top_dreg_dead_stats(d_reg, gate_dead, frac):
    flat_d = jnp.ravel(d_reg)
    flat_dead = jnp.ravel(gate_dead.astype(jnp.float32))
    total = flat_d.shape[0]
    k = max(1, int(total * frac))
    idx = jnp.argsort(-flat_d)[:k]
    dead_frac = jnp.mean(flat_dead[idx])
    pct = int(round(frac * 100))
    return {
        f"top_{pct}pct_dreg_dead_frac": _f(dead_frac),
        f"top_{pct}pct_dreg_preserved_frac": _f(1.0 - dead_frac),
    }


def _vlcp_pair_stats(
    l,
    weights,
    A_tildes,
    parent_trajectory,
    child_trajectory,
    target_kind,
    lambda_dyn,
    window,
    window_start,
    window_end,
    standardize_x,
    num_iters,
    dreg_zero_threshold,
    weight_dead_threshold,
    store_d_reg,
):
    """Run one VLCP layer-pair regression and summarize overlap stats."""
    d_reg, window_info = compute_dreg_for_pair(
        parent_trajectory,
        child_trajectory,
        window=window,
        window_start=window_start,
        window_end=window_end,
        lambda_dyn=lambda_dyn,
        num_iters=num_iters,
        standardize_x=standardize_x,
    )

    lasso_zero = d_reg < dreg_zero_threshold
    gate_dead = jnp.abs(weights[l]) < weight_dead_threshold
    overlap = jnp.logical_and(gate_dead, lasso_zero)

    gate_dead_count = jnp.sum(gate_dead.astype(jnp.float32))
    lasso_zero_count = jnp.sum(lasso_zero.astype(jnp.float32))
    overlap_count = jnp.sum(overlap.astype(jnp.float32))

    overlap_gate_to_lasso = jnp.where(
        gate_dead_count > 0.0,
        overlap_count / gate_dead_count,
        jnp.asarray(0.0),
    )
    overlap_lasso_to_gate = jnp.where(
        lasso_zero_count > 0.0,
        overlap_count / lasso_zero_count,
        jnp.asarray(0.0),
    )

    total_count = d_reg.size
    total_count_f = jnp.asarray(float(total_count), dtype=jnp.float32)
    expected_overlap_count = (
        gate_dead_count * lasso_zero_count / jnp.maximum(total_count_f, 1.0)
    )
    overlap_enrichment = jnp.where(
        expected_overlap_count > 0.0,
        overlap_count / expected_overlap_count,
        jnp.asarray(0.0),
    )

    d_abs = jnp.abs(d_reg)
    mean_dreg_gate_dead = _safe_masked_mean(d_abs, gate_dead)
    gate_alive = jnp.logical_not(gate_dead)
    mean_dreg_gate_alive = _safe_masked_mean(d_abs, gate_alive)
    dead_to_alive_dreg_ratio = jnp.where(
        mean_dreg_gate_alive > 1e-12,
        mean_dreg_gate_dead / (mean_dreg_gate_alive + 1e-12),
        jnp.asarray(0.0),
    )

    stats = {
        "l": int(l),
        "parent_dim": int(weights[l].shape[1]),
        "child_dim": int(weights[l].shape[0]),
        "target_kind": target_kind,
        "lambda_dyn": float(lambda_dyn),
        "window_start": int(window_info["window_start"]),
        "window_end": int(window_info["window_end"]),
        "window_steps": int(window_info["window_steps"]),
        "standardize_x": bool(window_info["standardize_x"]),
        "num_iters": int(num_iters),
        "d_reg_sparsity": _f(jnp.mean(lasso_zero.astype(jnp.float32))),
        "d_reg_nonzero_frac": _f(jnp.mean((~lasso_zero).astype(jnp.float32))),
        "d_reg_mean_abs": _f(jnp.mean(jnp.abs(d_reg))),
        "gate_dead_mask_sparsity": _f(jnp.mean(gate_dead.astype(jnp.float32))),
        "gate_alive_mask_sparsity": _f(jnp.mean(gate_alive.astype(jnp.float32))),
        "total_count": int(total_count),
        "gate_dead_count": int(_f(gate_dead_count)),
        "lasso_zero_count": int(_f(lasso_zero_count)),
        "overlap_count": int(_f(overlap_count)),
        "random_overlap_expected_count": _f(expected_overlap_count),
        "overlap_enrichment_vs_random": _f(overlap_enrichment),
        "overlap_gate_killed_to_lasso_zero": _f(overlap_gate_to_lasso),
        "overlap_lasso_zero_to_gate_killed": _f(overlap_lasso_to_gate),
        "mean_dreg_gate_dead": _f(mean_dreg_gate_dead),
        "mean_dreg_gate_alive": _f(mean_dreg_gate_alive),
        "dead_to_alive_dreg_ratio": _f(dead_to_alive_dreg_ratio),
        "corr_W_to_dreg": _f(_pearson_corr_flat(jnp.abs(weights[l]), d_reg)),
        "corr_Atilde_to_dreg": _f(
            _pearson_corr_flat(jnp.abs(A_tildes[l]), d_reg)
        ),
    }
    for frac in (0.01, 0.05, 0.10):
        stats.update(_top_dreg_dead_stats(d_reg, gate_dead, frac))
    if store_d_reg:
        stats["d_reg"] = jnp.asarray(d_reg).tolist()
    return stats


def _vlcp_trajectory_context(model, params, method, diag_X, diag_y):
    weights = _effective_weights_from_params(params)
    log_precisions = params["log_precisions"]
    lateral_U_list = params.get("lateral_U", [])
    lateral_log_alpha_list = params.get("lateral_log_alpha", [])
    num_layers = len(weights) + 1

    precisions = [jnp.exp(lp) for lp in log_precisions]
    lateral_pairs = _build_lateral_pairs_raw(lateral_U_list, lateral_log_alpha_list)

    num_classes = model.layer_sizes[-1]
    y_onehot = jnp.eye(num_classes)[diag_y]
    xs_init = _initialize_xs(weights, diag_X, y_onehot, num_layers)
    xs_eq, trajectory = infer_with_trajectory(
        weights,
        xs_init,
        precisions,
        lateral_pairs,
        method.num_inference_steps,
        method.lr_z,
    )
    return (
        weights,
        precisions,
        lateral_pairs,
        xs_eq,
        trajectory,
    )


def _vlcp_a_tildes(
    weights,
    precisions,
    lateral_pairs,
    xs_eq,
    method,
    A_tildes,
):
    if A_tildes is not None:
        return A_tildes

    jacobians = compute_jacobians(weights, xs_eq)
    output_fisher = compute_output_fisher(weights, xs_eq)
    return compute_local_maps(
        precisions,
        lateral_pairs,
        jacobians,
        method.ridge,
        output_fisher=output_fisher,
    )


def _vlcp_output_target_note():
    return (
        "Output state is label-clamped during inference; the output "
        "layer pair uses Delta logits rather than literal Delta z_output."
    )


def _vlcp_regression_from_trajectory(
    weights,
    A_tildes,
    trajectory,
    lambda_dyn,
    window,
    window_start,
    window_end,
    standardize_x,
    num_iters,
    dreg_zero_threshold,
    weight_dead_threshold,
    store_d_reg,
):
    layer_pairs = []
    for l in range(len(weights)):
        if l == len(weights) - 1:
            child_trajectory = _output_logit_trajectory(weights, trajectory)
            target_kind = "delta_logits"
        else:
            child_trajectory = trajectory[l + 1]
            target_kind = "delta_state"

        layer_pairs.append(
            _vlcp_pair_stats(
                l,
                weights,
                A_tildes,
                trajectory[l],
                child_trajectory,
                target_kind,
                lambda_dyn,
                window,
                window_start,
                window_end,
                standardize_x,
                num_iters,
                dreg_zero_threshold,
                weight_dead_threshold,
                store_d_reg,
            )
        )

    first_pair = layer_pairs[0] if layer_pairs else {}
    return {
        "lambda_dyn": float(lambda_dyn),
        "window": None if window is None else int(window),
        "window_start": first_pair.get("window_start"),
        "window_end": first_pair.get("window_end"),
        "window_steps": first_pair.get("window_steps"),
        "standardize_x": bool(standardize_x),
        "num_iters": int(num_iters),
        "dreg_zero_threshold": float(dreg_zero_threshold),
        "weight_dead_threshold": float(weight_dead_threshold),
        "store_d_reg": bool(store_d_reg),
        "output_target_note": _vlcp_output_target_note(),
        "layer_pairs": layer_pairs,
    }


def vlcp_regression_diagnostic(
    model,
    params,
    method,
    diag_X,
    diag_y,
    lambda_dyn=VLCP_LAMBDA_DYN_DEFAULT,
    window=VLCP_WINDOW_DEFAULT,
    window_start=None,
    window_end=None,
    standardize_x=False,
    num_iters=VLCP_NUM_ITERS_DEFAULT,
    dreg_zero_threshold=VLCP_DREG_ZERO_THRESHOLD,
    weight_dead_threshold=VLCP_WEIGHT_DEAD_THRESHOLD,
    store_d_reg=True,
    A_tildes=None,
):
    """VLCP regression diagnostic.

    Fits one sparse regression per child neuron over the inference trajectory
    and compares the resulting d_reg scores against gate-killed weights and
    Schur-Fisher A_tilde magnitudes.
    """
    (
        weights,
        precisions,
        lateral_pairs,
        xs_eq,
        trajectory,
    ) = _vlcp_trajectory_context(model, params, method, diag_X, diag_y)
    A_tildes = _vlcp_a_tildes(
        weights,
        precisions,
        lateral_pairs,
        xs_eq,
        method,
        A_tildes,
    )

    return _vlcp_regression_from_trajectory(
        weights,
        A_tildes,
        trajectory,
        lambda_dyn,
        window,
        window_start,
        window_end,
        standardize_x,
        num_iters,
        dreg_zero_threshold,
        weight_dead_threshold,
        store_d_reg,
    )


def vlcp_regression_sweep_diagnostic(
    model,
    params,
    method,
    diag_X,
    diag_y,
    lambdas=VLCP_SWEEP_LAMBDAS_DEFAULT,
    windows=VLCP_SWEEP_WINDOWS_DEFAULT,
    standardize_x_values=VLCP_SWEEP_STANDARDIZE_X_DEFAULT,
    num_iters=VLCP_NUM_ITERS_DEFAULT,
    dreg_zero_threshold=VLCP_DREG_ZERO_THRESHOLD,
    weight_dead_threshold=VLCP_WEIGHT_DEAD_THRESHOLD,
    store_d_reg=False,
    A_tildes=None,
):
    """Run the VLCP regression diagnostic over lambda/window/scaling sweeps."""
    (
        weights,
        precisions,
        lateral_pairs,
        xs_eq,
        trajectory,
    ) = _vlcp_trajectory_context(model, params, method, diag_X, diag_y)
    A_tildes = _vlcp_a_tildes(
        weights,
        precisions,
        lateral_pairs,
        xs_eq,
        method,
        A_tildes,
    )

    configs = []
    windows_to_run = list(windows)

    for lambda_dyn in lambdas:
        for window_start, window_end in windows_to_run:
            for standardize_x in standardize_x_values:
                config = _vlcp_regression_from_trajectory(
                    weights,
                    A_tildes,
                    trajectory,
                    lambda_dyn,
                    None,
                    window_start,
                    window_end,
                    standardize_x,
                    num_iters,
                    dreg_zero_threshold,
                    weight_dead_threshold,
                    store_d_reg,
                )
                config["config_index"] = int(len(configs))
                config.pop("output_target_note", None)
                configs.append(config)

    return {
        "diagnostic_kind": "vlcp_regression_sweep",
        "sweep_lambdas": [float(x) for x in lambdas],
        "sweep_windows": [
            {"window_start": int(start), "window_end": int(end)}
            for start, end in windows_to_run
        ],
        "sweep_standardize_x": [bool(x) for x in standardize_x_values],
        "num_configs": int(len(configs)),
        "num_iters": int(num_iters),
        "dreg_zero_threshold": float(dreg_zero_threshold),
        "weight_dead_threshold": float(weight_dead_threshold),
        "store_d_reg": bool(store_d_reg),
        "output_target_note": _vlcp_output_target_note(),
        "configs": configs,
    }


def run_full_diagnostics(
    model, params, method,
    diag_X, diag_y,
    train_X, train_y,
    test_X, test_y,
    label,
    include_structural=False,
    include_vlcp=False,
    include_vlcp_sweep=False,
):
    """Run all diagnostic levels and return a nested dict."""
    weights = _effective_weights_from_params(params)
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
    inf_diag, xs_eq = inference_trajectory(
        weights, xs_init, precisions, lateral_pairs,
        method.num_inference_steps, method.lr_z,
    )

    # Equilibrium state used by the remaining diagnostics.
    errors = compute_errors(weights, xs_eq)

    # Causal influence maps and gates.
    output_fisher = compute_output_fisher(weights, xs_eq)
    cc_diag, jacobians, A_tildes, gates = cc_machinery_stats(
        weights, xs_eq, precisions, lateral_pairs,
        method.gate_p, method.gate_kappa, method.ridge,
        output_fisher=output_fisher,
    )
    of_diag = output_fisher_stats(output_fisher, method.ridge)

    # Weight-update magnitudes.
    wu_diag = weight_update_stats(
        weights,
        xs_eq,
        errors,
        gates,
        precisions,
        diag_X.shape[0],
    )

    # Precision and residual statistics. Pass d_min/d_max from the method so
    # the structured-D block reports clip-band proximity even when the structured
    # path is disabled (D ≡ 1 there).
    pi_diag = precision_stats(
        log_precisions, errors, precision_var_ema,
        d_min=getattr(method, "d_min", None),
        d_max=getattr(method, "d_max", None),
    )

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

    if "vertical_gate_logits" in params and "vertical_importance" in params:
        out["vertical_pruning"] = vertical_pruning_stats(
            params["vertical_gate_logits"],
            params["vertical_importance"],
            params.get(
                "vertical_layer_scales",
                [jnp.asarray(0.0) for _ in params["vertical_gate_logits"]],
            ),
            prune_threshold=getattr(method, "vertical_prune_threshold", 0.1),
            match_loss=params.get("vertical_match_loss", 0.0),
            sparse_loss=params.get("vertical_sparse_loss", 0.0),
        )

    # Optional comparison between feedforward states and inferred states.
    if include_structural:
        out["structural"] = equilibrium_vs_feedforward_gap(
            model, params, method, diag_X, diag_y
        )

    # Optional VLCP regression diagnostic. This is intentionally
    # opt-in because it fits one LASSO per child neuron and is meant for the
    # post-training full diagnostic, not per-epoch snapshots.
    if include_vlcp_sweep:
        out["vlcp_regression"] = vlcp_regression_sweep_diagnostic(
            model,
            params,
            method,
            diag_X,
            diag_y,
            A_tildes=A_tildes,
        )
    elif include_vlcp:
        out["vlcp_regression"] = vlcp_regression_diagnostic(
            model,
            params,
            method,
            diag_X,
            diag_y,
            A_tildes=A_tildes,
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
        f"L{l} mean={c['gate_mean']:.3e} max={c['gate_max']:.3f} "
        f"frac>2x={c['frac_above_2x_mean']:.3f}"
        for l, c in enumerate(cc)
    )
    print(f"  Gates:         {gate_str}")

    ent_str = "   ".join(
        f"L{l} row_H={c['row_entropy_mean']:.2f}/{c['row_entropy_max_possible']:.2f}"
        for l, c in enumerate(cc)
    )
    print(f"  Gate entropy:  {ent_str}")

    rowsum_str = "   ".join(
        f"L{l} mean={c['row_sum_mean']:.3f} [min={c['row_sum_min']:.3f}, max={c['row_sum_max']:.3f}]"
        for l, c in enumerate(cc)
    )
    print(f"  Gate row-sum:  {rowsum_str}")

    if cc and "per_sample_A_std_over_mean" in cc[0]:
        ps_str = "   ".join(
            f"L{l} σ/μ={c['per_sample_A_std_over_mean']:.3f} "
            f"Δbatch={c['A_batch_averaged_vs_persample_diff']:.3f}"
            for l, c in enumerate(cc)
        )
        print(f"  Per-sample Ã:  {ps_str}")

    if cc and "per_sample_gate_agreement" in cc[0]:
        pg_str = "   ".join(
            f"L{l} agree={c['per_sample_gate_agreement']:.3f} "
            f"max={c['per_sample_gate_max_mean']:.3f} "
            f"row_H={c['per_sample_row_H_mean']:.2f}"
            for l, c in enumerate(cc)
        )
        print(f"  Per-sample G:  {pg_str}")

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
        # D block. Always shown for hidden layers; when the
        # structured path is off, D collapses to 1 by construction.
        if "D_min" in x and "D_max" in x:
            chunk += f" D=[{x['D_min']:.2f},{x['D_max']:.2f}]"
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

    vert = diag.get("vertical_pruning", {}).get("layers", [])
    if vert:
        vert_parts = []
        for x in vert:
            vert_parts.append(
                f"L{x['l']} scale={x['layer_scale']:.2f} "
                f"G={x['gate_mean']:.3f} eff_min={x['effective_gate_min']:.3f} "
                f"pruned={x['effective_pruned_frac'] * 100:.1f}%"
            )
        losses = diag.get("vertical_pruning", {})
        print(
            f"  Vert prune:    {'   '.join(vert_parts)}   "
            f"Lm={losses.get('match_loss', 0.0):.2e} "
            f"Ls={losses.get('sparse_loss', 0.0):.2e}"
        )

    vlcp_block = diag.get("vlcp_regression", {})
    vlcp = vlcp_block.get("layer_pairs", [])
    if vlcp:
        vlcp_parts = []
        for x in vlcp:
            vlcp_parts.append(
                f"L{x['l']} kill_to_zero={x['overlap_gate_killed_to_lasso_zero']:.3f} "
                f"enrich={x.get('overlap_enrichment_vs_random', float('nan')):.3f} "
                f"A~corr={x['corr_Atilde_to_dreg']:.3f}"
            )
        print(f"  VLCP reg:      {'   '.join(vlcp_parts)}")
    elif "configs" in vlcp_block:
        best = None
        for config in vlcp_block["configs"]:
            for pair in config.get("layer_pairs", []):
                value = pair.get("overlap_enrichment_vs_random")
                if value is not None and (best is None or value > best[0]):
                    best = (value, config, pair)
        if best is None:
            print(f"  VLCP sweep:    configs={vlcp_block.get('num_configs', 0)}")
        else:
            value, config, pair = best
            print(
                f"  VLCP sweep:    configs={vlcp_block.get('num_configs', 0)} "
                f"best=L{pair['l']} enrich={value:.3f} "
                f"lambda={config['lambda_dyn']:.0e} "
                f"win={config['window_start']}:{config['window_end']} "
                f"std={config['standardize_x']}"
            )

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
