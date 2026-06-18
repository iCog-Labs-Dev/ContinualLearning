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

model = CausalCodingModel([784, 512, 512, 10], lateral_init_scale=0.01)
key = jax.random.PRNGKey(0)
params = model.init_params(key)

method = CausalCodingMethod(
    lr_z=0.01,
    lr_w=0.2,
    lr_pi=0.0001,
    num_inference_steps=25,
    gate_p=2.0,
    gate_kappa=1e-3,
    ridge=1e-4,
    lambda_s=1e-6,
    batch_size=128,
    epochs=25,
)

CLBenchmark(method=method, model=model, tasks=tasks, name="causal_coding").run(
    params, None
)
