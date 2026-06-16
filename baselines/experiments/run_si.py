import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.base import SIState
from benchmarker import CLBenchmark
from src.si import SIMethod

X, y, test_X, test_y = load_mnist()
class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)

model = MLP([784, 512, 512, 10])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

method = SIMethod(
    lr=0.001,
    lr_task1=0.01,
    batch_size=128,
    epochs=25,
    lam=500.0,
    normalize=True,
)
state = SIState(
    old_params=params,
    cumulative_omega=jax.tree.map(lambda p: jnp.zeros_like(p), params),
)

CLBenchmark(method=method, model=model, tasks=tasks, name="si").run(params, state)
