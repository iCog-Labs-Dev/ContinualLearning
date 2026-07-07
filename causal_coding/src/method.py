import jax.numpy as jnp

from core.data import Task
from causal_coding.src.lateral import ramp_schedule
from causal_coding.src.training import train_step
from causal_coding.src.vertical_pruning import (
    apply_vertical_gates,
    compute_regression_importance,
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
        num_classes = model.layer_sizes[-1]
        y_onehot = jnp.eye(num_classes)[task.train_y]
        num_batch = task.train_X.shape[0] // self.batch_size

        # Task-IL uses an independent-Bernoulli output edge restricted to the
        # task's active classes. Class-IL uses the full categorical softmax
        # head.
        task_il_training = getattr(self, "task_il_training", False)
        if task_il_training:
            active_mask = jnp.zeros(num_classes).at[jnp.array(task.classes)].set(1.0)
        else:
            active_mask = jnp.ones(num_classes)

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
                    self.lambda_d,
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
                )
                total_loss += loss

            epoch_loss = total_loss / num_batch
            print(
                f"Epoch {ep + 1} ========= "
                f"Loss: {epoch_loss}  "
                f"[ramp={lateral_force_scale:.2f}, "
                f"lr_scale={lateral_lr_scale:.1f}, "
                f"vert={vertical_lr_scale:.2f}]"
            )

            if diagnostics_hook is not None:
                diagnostics_hook(params, ep, epoch_loss)

        return params, state, total_loss / num_batch
