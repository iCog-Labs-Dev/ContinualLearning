import jax.numpy as jnp

from core.data import Task
from causal_coding.src.lateral import ramp_schedule
from causal_coding.src.training import train_step


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
        gate_alpha=0.6,
        gate_floor_coeff=0.05,
        gate_warmup_epochs=0,
        gate_ramp_epochs=0,
        beta_pi=0.99,
        epsilon_pi=1e-4,
        alpha_pi=0.05,
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
    ):
        self.lr_z = lr_z
        self.lr_w = lr_w
        self.num_inference_steps = num_inference_steps
        self.gate_p = gate_p
        self.gate_kappa = gate_kappa
        self.ridge = ridge
        self.lambda_s = lambda_s
        self.batch_size = batch_size
        self.epochs = epochs
        self.gate_alpha = gate_alpha
        self.gate_floor_coeff = gate_floor_coeff
        self.gate_warmup_epochs = gate_warmup_epochs
        self.gate_ramp_epochs = gate_ramp_epochs

        # Effective gate strength used by diagnostics during training.
        self.gate_alpha_current = gate_alpha
        self.beta_pi = beta_pi
        self.epsilon_pi = epsilon_pi
        self.alpha_pi = alpha_pi
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

    def train_task(self, model, params, state, task: Task, task_idx, diagnostics_hook=None):
        num_classes = model.layer_sizes[-1]
        y_onehot = jnp.eye(num_classes)[task.train_y]
        num_batch = task.train_X.shape[0] // self.batch_size

        for ep in range(self.epochs):

            if task_idx == 0:
                lateral_force_scale = ramp_schedule(
                    ep, self.lat_warmup_epochs, self.lat_ramp_epochs
                )
                lateral_lr_scale = 1.0 if ep >= self.lat_warmup_epochs else 0.0
                gate_alpha_scale = ramp_schedule(
                    ep, self.gate_warmup_epochs, self.gate_ramp_epochs
                )
            else:
                lateral_force_scale = 1.0
                lateral_lr_scale = 1.0
                gate_alpha_scale = 1.0
            gate_alpha_eff = gate_alpha_scale * self.gate_alpha
            self.gate_alpha_current = gate_alpha_eff

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
                    gate_alpha_eff,
                    self.gate_floor_coeff,
                    self.beta_pi,
                    self.epsilon_pi,
                    self.alpha_pi,
                    # Epoch-dependent training schedules.
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
                )
                total_loss += loss

            epoch_loss = total_loss / num_batch
            print(
                f"Epoch {ep + 1} ========= "
                f"Loss: {epoch_loss}  "
                f"[gate_α_eff={gate_alpha_eff:.3f}, "
                f"ramp={lateral_force_scale:.2f}, lr_scale={lateral_lr_scale:.1f}]"
            )

            if diagnostics_hook is not None:
                diagnostics_hook(params, ep, epoch_loss)

        return params, state, total_loss / num_batch
