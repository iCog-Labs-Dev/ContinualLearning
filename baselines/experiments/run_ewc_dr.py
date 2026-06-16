import sys
import os
import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.base import EWCVanillaState
from benchmarker import CLBenchmark
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

CLBenchmark(method=method, model=model, tasks=tasks, name="ewc_dr").run(params, state)
