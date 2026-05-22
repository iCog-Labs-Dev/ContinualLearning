import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.base import EWCVanillaState
from core.runner import evaluate, run_experiment
from benchmarker import benchmark
from src.ewc_dr import EWCDRMethod

X, y, test_X, test_y = load_mnist()
class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)

model = MLP([784, 512, 512, 10])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

method = EWCDRMethod(
    lr=0.001, lr_task1=0.01, batch_size=128, epochs=25, lam=100, num_samples=200
)
state = EWCVanillaState(anchors=[])

class_il_baselines = [evaluate(model, params, task) for task in tasks]
task_il_baselines = [evaluate(model, params, task, task.classes) for task in tasks]

params, _, class_il_matrix, task_il_matrix = run_experiment(
    method, model, params, state, tasks
)

benchmark("ewc_dr", class_il_matrix, task_il_matrix, class_il_baselines, task_il_baselines)
