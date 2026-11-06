import jax
import jax.numpy as jnp

from core.data import Task
from causal_coding.src.lateral import ramp_schedule
from causal_coding.src.training import train_step
from causal_coding.src.vertical_pruning import (
    apply_vertical_gates,
    compute_regression_importance,
)
from causal_coding.src.meta_learner import (
    MetaState,
    init_meta_state,
    decode_omega,
    update_drift_and_detect_shift,
    calibrate_beta_via_intervention,
    compute_locality_and_commutator_proxies,
    compute_pairwise_confusion_matrix,
    sample_anchor_batch,
    meta_step,
)


class CausalCodingMethod:
    def __init__(
        self,
        lr_z,
        lr_w,
        num_inference_steps,
        gate_p,
        gate_kappa,
        ridge,
        lambda_s,
        batch_size,
        epochs,
        beta_pi=0.99,
        k_probe=5,

        # Low-rank PSD hidden lateral precision.
        lr_lat=1e-5,
        beta_cov=0.99,
        eps_lat=1e-2,
        lat_warmup_epochs=5,
        lat_ramp_epochs=10,
        lambda_max_cap=0.3,
        beta_logdet=1e-3,
        lambda_fro=1e-3,
        lambda_U=1e-5,
        adam_beta1_lat=0.9,
        adam_beta2_lat=0.999,
        adam_eps_lat=1e-8,
        grad_clip_norm_lat=1.0,

        # Diffusion clarity penalty.
        lambda_d=0.0,
        clarity_t=1.0,
        clarity_eps=1e-4,

        # Structured diagonal residual precision.
        # Hidden log_precisions use log(pi0) + log(D_l), where D_l is a
        # mean-1 per-unit relative scaling clipped to [d_min, d_max].
        pi0=2.718281828459045,
        rho_v=0.1,
        delta_abs=1e-12,
        d_min=0.5,
        d_max=2.0,

        # Vertical soft-pruning gates.
        vertical_pruning_enabled=False,
        vertical_layer_scales=(1.0, 1.0, 0.0),
        vertical_alpha_g=8.0,
        lambda_vert_match=3e-3,
        lambda_vert_sparse=5e-5,
        vertical_eps=1e-2,
        lr_vert=1e-3,
        vertical_warmup_epochs=30,
        vertical_ramp_epochs=10,
        vertical_importance_update_epochs=5,
        vertical_importance_batch_size=256,
        vertical_prune_threshold=0.1,
        vertical_importance_lambda_dyn=1e-3,
        vertical_importance_window=15,
        vertical_importance_num_iters=50,
        vertical_importance_standardize_x=False,
        adam_beta1_vert=0.9,
        adam_beta2_vert=0.999,
        adam_eps_vert=1e-8,
        grad_clip_norm_vert=1.0,
        gate_norm="sum",

        # Meta-learning settings (Goertzel 2025)
        meta_learning_enabled=False,
        lr_meta=1e-2,
        meta_beta_lcb_init=1.0,
        meta_tau_init=1e-3,
        meta_eps_exp_init=0.05,
        meta_protection_init=0.8,
        anchor_size_per_task=64,
        lambda_anchor=0.2,
        meta_alpha_forget=1.0,
        meta_beta_confusion=0.5,
        meta_calibrate_every_epochs=5,
        fixed_controller=False,
        disable_exploration=False,
    ):
        self.lr_z = lr_z
        self.lr_w = lr_w
        self.num_inference_steps = num_inference_steps
        self.gate_p = gate_p
        self.gate_kappa = gate_kappa
        # Causal-gate magnitude normalization: "sum" keeps row sums near 1;
        # "match" preserves each row's update norm after gating.
        self.gate_norm = gate_norm
        self.ridge = ridge
        self.lambda_s = lambda_s
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta_pi = beta_pi
        self.k_probe = k_probe

        # Lateral precision optimization.
        self.lr_lat = lr_lat
        self.beta_cov = beta_cov
        self.eps_lat = eps_lat
        self.lat_warmup_epochs = lat_warmup_epochs
        self.lat_ramp_epochs = lat_ramp_epochs
        self.lambda_max_cap = lambda_max_cap
        self.beta_logdet = beta_logdet
        self.lambda_fro = lambda_fro
        self.lambda_U = lambda_U
        self.adam_beta1_lat = adam_beta1_lat
        self.adam_beta2_lat = adam_beta2_lat
        self.adam_eps_lat = adam_eps_lat
        self.grad_clip_norm_lat = grad_clip_norm_lat

        # Diffusion clarity penalty.
        self.lambda_d = lambda_d
        self.clarity_t = clarity_t
        self.clarity_eps = clarity_eps

        # Structured diagonal residual precision.
        self.pi0 = pi0
        self.rho_v = rho_v
        self.delta_abs = delta_abs
        self.d_min = d_min
        self.d_max = d_max

        # Vertical soft-pruning gates.
        self.vertical_pruning_enabled = vertical_pruning_enabled
        self.vertical_layer_scales = tuple(vertical_layer_scales)
        self.vertical_alpha_g = vertical_alpha_g
        self.lambda_vert_match = lambda_vert_match
        self.lambda_vert_sparse = lambda_vert_sparse
        self.vertical_eps = vertical_eps
        self.lr_vert = lr_vert
        self.vertical_warmup_epochs = vertical_warmup_epochs
        self.vertical_ramp_epochs = vertical_ramp_epochs
        self.vertical_importance_update_epochs = vertical_importance_update_epochs
        self.vertical_importance_batch_size = vertical_importance_batch_size
        self.vertical_prune_threshold = vertical_prune_threshold
        self.vertical_importance_lambda_dyn = vertical_importance_lambda_dyn
        self.vertical_importance_window = vertical_importance_window
        self.vertical_importance_num_iters = vertical_importance_num_iters
        self.vertical_importance_standardize_x = vertical_importance_standardize_x
        self.adam_beta1_vert = adam_beta1_vert
        self.adam_beta2_vert = adam_beta2_vert
        self.adam_eps_vert = adam_eps_vert
        self.grad_clip_norm_vert = grad_clip_norm_vert
        self.vertical_layer_scales_current = ()

        # Meta-learning attributes
        self.meta_learning_enabled = meta_learning_enabled
        self.lr_meta = lr_meta
        self.meta_beta_lcb_init = meta_beta_lcb_init
        self.meta_tau_init = meta_tau_init
        self.meta_eps_exp_init = meta_eps_exp_init
        self.meta_protection_init = meta_protection_init
        self.anchor_size_per_task = anchor_size_per_task
        self.lambda_anchor = lambda_anchor
        self.meta_alpha_forget = meta_alpha_forget
        self.meta_beta_confusion = meta_beta_confusion
        self.meta_calibrate_every_epochs = meta_calibrate_every_epochs
        self.fixed_controller = fixed_controller
        self.disable_exploration = disable_exploration
        if self.disable_exploration:
            self.meta_eps_exp_init = 0.0

    def _configured_vertical_layer_scales(self, num_weight_layers):
        if len(self.vertical_layer_scales) == num_weight_layers:
            return list(self.vertical_layer_scales)
        if self.vertical_layer_scales == (1.0, 1.0, 0.0):
            return [1.0] * max(num_weight_layers - 1, 0) + [0.0]
        raise ValueError(
            "vertical_layer_scales length must match the number of weight layers "
            f"({num_weight_layers}), got {len(self.vertical_layer_scales)}"
        )

    def _active_vertical_layer_scales(self, num_weight_layers, vertical_ramp_scale):
        configured = self._configured_vertical_layer_scales(num_weight_layers)
        if not self.vertical_pruning_enabled:
            return [0.0 for _ in configured]
        return [vertical_ramp_scale * s for s in configured]

    def _with_vertical_layer_scales(self, params, layer_scales):
        new_params = dict(params)
        dtype = params["weights"][0].dtype
        new_params["vertical_layer_scales"] = [
            jnp.asarray(scale, dtype=dtype) for scale in layer_scales
        ]
        return new_params

    def _refresh_vertical_importance(
        self,
        params,
        task,
        y_onehot,
        lateral_force_scale,
        layer_scales,
        task_il_training=False,
        active_mask=None,
    ):
        n = min(int(self.vertical_importance_batch_size), int(task.train_X.shape[0]))
        x_imp = task.train_X[:n]
        y_imp = y_onehot[:n]
        effective_weights = apply_vertical_gates(
            params["weights"],
            params["vertical_gate_logits"],
            layer_scales,
        )
        importance = compute_regression_importance(
            effective_weights,
            x_imp,
            y_imp,
            params["log_precisions"],
            params["lateral_U"],
            params["lateral_log_alpha"],
            lateral_force_scale,
            self.num_inference_steps,
            self.lr_z,
            lambda_dyn=self.vertical_importance_lambda_dyn,
            window=self.vertical_importance_window,
            num_iters=self.vertical_importance_num_iters,
            standardize_x=self.vertical_importance_standardize_x,
            task_il_training=task_il_training,
            active_mask=active_mask,
        )
        new_params = dict(params)
        new_params["vertical_importance"] = importance
        return new_params

    def train_task(self, model, params, state, task: Task, task_idx, diagnostics_hook=None):
        import time
        import tracemalloc

        t_task_start = time.time()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        tracemalloc.reset_peak()

        num_classes = model.layer_sizes[-1]
        y_onehot = jnp.eye(num_classes)[task.train_y]
        num_batch = max(1, task.train_X.shape[0] // self.batch_size)

        # Task-IL uses an independent-Bernoulli output edge restricted to the
        # task's active classes. Class-IL uses the full categorical softmax
        # head.
        task_il_training = getattr(self, "task_il_training", False)
        if task_il_training:
            active_mask = jnp.zeros(num_classes).at[jnp.array(task.classes)].set(1.0)
        else:
            active_mask = jnp.ones(num_classes)

        if self.meta_learning_enabled:
            if state is None or task_idx == 0:
                state = init_meta_state(
                    params,
                    beta_lcb_init=self.meta_beta_lcb_init,
                    tau_init=self.meta_tau_init,
                    eps_exp_init=self.meta_eps_exp_init,
                    lambda_d_init=self.lambda_d if self.lambda_d > 0.0 else 1e-3,
                    protection_init=self.meta_protection_init,
                )
            else:
                task_class_mask = jnp.zeros(num_classes).at[jnp.array(task.classes)].set(1.0)
                state.consolidated_support_mask[-1] = (
                    state.consolidated_support_mask[-1] * (1.0 - task_class_mask[:, None])
                )
                if "vertical_gate_logits" in params and params["vertical_gate_logits"] is not None:
                    params["vertical_gate_logits"] = [
                        jnp.where(
                            state.consolidated_support_mask[l] > 0.5,
                            params["vertical_gate_logits"][l],
                            4.595,
                        )
                        for l in range(len(params["vertical_gate_logits"]))
                    ]
            weights_start = [jnp.array(w) for w in params["weights"]]

        for ep in range(self.epochs):
            if task_idx == 0:
                lateral_force_scale = ramp_schedule(
                    ep, self.lat_warmup_epochs, self.lat_ramp_epochs
                )
                lateral_lr_scale = 1.0 if ep >= self.lat_warmup_epochs else 0.0
            else:
                lateral_force_scale = 1.0
                lateral_lr_scale = 1.0

            vertical_ramp_scale = ramp_schedule(
                ep, self.vertical_warmup_epochs, self.vertical_ramp_epochs
            )
            vertical_lr_scale = (
                vertical_ramp_scale if self.vertical_pruning_enabled else 0.0
            )
            vertical_layer_scales = self._active_vertical_layer_scales(
                len(params["weights"]), vertical_ramp_scale
            )
            self.vertical_layer_scales_current = tuple(vertical_layer_scales)
            params = self._with_vertical_layer_scales(params, vertical_layer_scales)

            if (
                self.vertical_pruning_enabled
                and vertical_lr_scale > 0.0
                and self.vertical_importance_update_epochs > 0
                and (ep - self.vertical_warmup_epochs)
                % self.vertical_importance_update_epochs
                == 0
            ):
                params = self._refresh_vertical_importance(
                    params,
                    task,
                    y_onehot,
                    lateral_force_scale,
                    vertical_layer_scales,
                    task_il_training=task_il_training,
                    active_mask=active_mask,
                )

            if self.meta_learning_enabled:
                (
                    meta_beta_lcb,
                    meta_tau,
                    meta_eps_exp,
                    meta_lambda_d,
                    meta_protection,
                ) = decode_omega(state.omega)
                if self.disable_exploration:
                    meta_eps_exp = 0.0
                eff_lambda_d = meta_lambda_d
            else:
                eff_lambda_d = self.lambda_d
                meta_beta_lcb = 1.0
                meta_tau = 1e-3
                meta_eps_exp = 0.05
                meta_protection = 0.0

            total_loss = 0.0
            for i in range(num_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                batch_X = task.train_X[start:end]
                batch_y = y_onehot[start:end]

                params, loss = train_step(
                    params,
                    batch_X,
                    batch_y,
                    self.num_inference_steps,
                    self.k_probe,
                    self.lr_z,
                    self.lr_w,
                    self.gate_p,
                    self.gate_kappa,
                    self.ridge,
                    self.lambda_s,
                    self.beta_pi,
                    lateral_force_scale,
                    lateral_lr_scale,
                    self.lr_lat,
                    self.beta_cov,
                    self.eps_lat,
                    self.beta_logdet,
                    self.lambda_fro,
                    self.lambda_U,
                    self.lambda_max_cap,
                    self.adam_beta1_lat,
                    self.adam_beta2_lat,
                    self.adam_eps_lat,
                    self.grad_clip_norm_lat,
                    eff_lambda_d,
                    self.clarity_t,
                    self.clarity_eps,
                    vertical_lr_scale,
                    vertical_layer_scales,
                    self.lr_vert,
                    self.vertical_alpha_g,
                    self.lambda_vert_match,
                    self.lambda_vert_sparse,
                    self.vertical_eps,
                    self.adam_beta1_vert,
                    self.adam_beta2_vert,
                    self.adam_eps_vert,
                    self.grad_clip_norm_vert,
                    self.pi0,
                    self.rho_v,
                    self.delta_abs,
                    self.d_min,
                    self.d_max,
                    active_mask,
                    task_il_training,
                    self.gate_norm,
                    self.meta_learning_enabled,
                    meta_beta_lcb,
                    meta_tau,
                    meta_eps_exp,
                    meta_protection,
                    state.consolidated_support_mask if (self.meta_learning_enabled and task_idx > 0) else None,
                )
                total_loss += loss

            epoch_loss = total_loss / num_batch

            if self.meta_learning_enabled:
                last_mu = params["last_mu_influence"]
                last_sigma = params["last_sigma_influence"]
                last_support = params["last_support_masks"]
                last_eps_loc = params["last_eps_locality"]

                state, drift_stat, is_shift = update_drift_and_detect_shift(state, last_mu)

                if not self.fixed_controller:
                    if (ep + 1) % self.meta_calibrate_every_epochs == 0:
                        probe_n = min(32, task.train_X.shape[0])
                        state = calibrate_beta_via_intervention(
                            model,
                            params,
                            state,
                            task.train_X[:probe_n],
                            y_onehot[:probe_n],
                            last_mu,
                            last_sigma,
                            active_classes=task.classes if task_il_training else None,
                        )

                    state = meta_step(
                        model,
                        params,
                        state,
                        batch_X,
                        batch_y,
                        active_mask,
                        task_il_training,
                        self.lr_w,
                        self.lambda_s,
                        self.gate_p,
                        self.gate_kappa,
                        gate_norm=self.gate_norm,
                        alpha_forget=self.meta_alpha_forget,
                        beta_confusion=self.meta_beta_confusion,
                        lr_meta=self.lr_meta,
                        step_seed=ep * num_batch + (num_batch - 1),
                    )

                # Compute per-layer gating, UCB exploration, and reactivation diagnostics
                layer_diags = []
                for l in range(len(last_mu)):
                    mu_l = last_mu[l]
                    sig_l = last_sigma[l]
                    sup_l = last_support[l]
                    lcb_l = jax.nn.relu(mu_l - meta_beta_lcb * sig_l - meta_tau)
                    ucb_l = jax.nn.relu(mu_l + meta_beta_lcb * sig_l - meta_tau)
                    exp_mask = (lcb_l <= 0.0) & (ucb_l > 0.0)
                    exp_rate = float(jnp.mean(exp_mask.astype(jnp.float32)))
                    sup_dens = float(jnp.mean((sup_l > 0.0).astype(jnp.float32)))
                    if len(state.task_supports) > 0 and state.consolidated_support_mask[l] is not None:
                        past_sup = state.consolidated_support_mask[l] > 0.0
                        react_rate = float(jnp.mean((past_sup & exp_mask).astype(jnp.float32)))
                    else:
                        react_rate = 0.0
                    layer_diags.append({
                        "layer": int(l),
                        "exploratory_active_rate": exp_rate,
                        "support_density": sup_dens,
                        "reactivated_rate": react_rate,
                    })

                state.meta_trajectory.append({
                    "epoch": int(ep + 1),
                    "task": int(task_idx),
                    "beta_lcb": float(meta_beta_lcb),
                    "tau": float(meta_tau),
                    "eps_exp": float(meta_eps_exp),
                    "lambda_d": float(eff_lambda_d),
                    "protection": float(meta_protection),
                    "eps_locality": float(last_eps_loc),
                    "drift_D": float(drift_stat),
                    "loss": float(epoch_loss),
                    "gate_align_cos_sim": float(getattr(state, "last_gate_align_cos_sim", 1.0)),
                    "gate_align_rel_err": float(getattr(state, "last_gate_align_rel_err", 0.0)),
                    "layer_diagnostics": layer_diags,
                })

                print(
                    f"Epoch {ep + 1} ========= "
                    f"Loss: {epoch_loss:.5f}  "
                    f"[beta_lcb={float(meta_beta_lcb):.3f}, tau={float(meta_tau):.4f}, "
                    f"eps_exp={float(meta_eps_exp):.3f}, lambda_d={float(eff_lambda_d):.4f}, "
                    f"eps_loc={float(last_eps_loc):.4f}, drift_D={float(drift_stat):.4f}]"
                )
            else:
                print(
                    f"Epoch {ep + 1} ========= "
                    f"Loss: {epoch_loss}  "
                    f"[ramp={lateral_force_scale:.2f}, "
                    f"lr_scale={lateral_lr_scale:.1f}, "
                    f"vert={vertical_lr_scale:.2f}]"
                )

            if diagnostics_hook is not None:
                diagnostics_hook(params, ep, epoch_loss)

        if self.meta_learning_enabled:
            task_class_mask = jnp.zeros(num_classes).at[jnp.array(task.classes)].set(1.0)
            task_support_mask = list(params["last_support_masks"])
            task_support_mask[-1] = task_support_mask[-1] * task_class_mask[:, None]
            state.task_supports.append(task_support_mask)

            delta_W = [
                w_end - w_start
                for w_end, w_start in zip(params["weights"], weights_start)
            ]
            norm_dw = jnp.sqrt(sum(jnp.sum(dw ** 2) for dw in delta_W)) + 1e-8
            norm_dir = [dw / norm_dw for dw in delta_W]
            state.task_update_dirs.append(norm_dir)

            # Fix 3: Compute and log the true T x T pairwise confusion matrix delta_hat_{c,c'}
            state.confusion_graph = compute_pairwise_confusion_matrix(
                state.task_supports,
                state.task_update_dirs,
                state.consolidated_support_mask,
            )

            state.consolidated_support_mask = [
                jnp.maximum(old, new)
                for old, new in zip(state.consolidated_support_mask, task_support_mask)
            ]

            n_anc = min(self.anchor_size_per_task, task.train_X.shape[0])
            anc_key = jax.random.PRNGKey(task_idx * 1000 + 42)
            anc_indices = jax.random.permutation(anc_key, task.train_X.shape[0])[:n_anc]
            state.anchor_buffer.append({
                "X": task.train_X[anc_indices],
                "y": y_onehot[anc_indices],
                "classes": task.classes,
                "task_idx": task_idx,
            })
            self.last_meta_state = state

            t_task_elapsed = time.time() - t_task_start
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_memory_mb = peak_bytes / (1024 * 1024)

            if not hasattr(state, "task_overhead"):
                state.task_overhead = []
            state.task_overhead.append({
                "task": int(task_idx),
                "duration_seconds": float(t_task_elapsed),
                "peak_memory_mb": float(peak_memory_mb),
            })

            protocol_key = "task_il" if task_il_training else "class_il"
            if not hasattr(self, "meta_diagnostics"):
                self.meta_diagnostics = {}
            self.meta_diagnostics[protocol_key] = {
                "trajectory": list(state.meta_trajectory),
                "drift_history": list(state.drift_history),
                "confusion_graph": list(state.confusion_graph),
                "task_overhead": list(state.task_overhead),
            }

        return params, state, total_loss / num_batch
