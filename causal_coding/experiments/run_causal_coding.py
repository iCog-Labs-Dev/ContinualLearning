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
    lr_w=0.02,
    num_inference_steps=30,
    gate_p=2.0,
    gate_kappa=1e-3,
    ridge=1e-4,
    lambda_s=1e-6,
    batch_size=128,
    epochs=50,
    gate_alpha=0.3,
    gate_floor_coeff=0.05,
    gate_warmup_epochs=15,
    gate_ramp_epochs=15,
    beta_pi=0.99,
    epsilon_pi=0.1,
    alpha_pi=0.05,
    k_probe=10,
    # Lateral precision optimization.
    lr_lat=1e-3,
    beta_cov=0.99,
    eps_lat=1e-2,
    lat_warmup_epochs=5,
    lat_ramp_epochs=10,
    lambda_max_cap=1.0,
    beta_logdet=0.1,
    lambda_fro=1e-2,
    lambda_U=1e-5,
)

CLBenchmark(method=method, model=model, tasks=tasks, name="causal_coding").run(
    params, None
)
