import sys
import os
import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.data import load_mnist, split_into_tasks
from benchmarker import CLBenchmark
from src.model import CausalCodingModel
from src.method import CausalCodingMethod

X, y, test_X, test_y = load_mnist()
class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)

model = CausalCodingModel(
    [784, 512, 512, 10],
    lateral_rank=32,
    lateral_init_scale=1e-3,
    alpha_init=0.01,
)
key = jax.random.PRNGKey(0)
params = model.init_params(key)

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

    # Tight-clipped structured diagonal residual precision.
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
)

CLBenchmark(method=method, model=model, tasks=tasks, name="causal_coding").run(
    params, None
)
