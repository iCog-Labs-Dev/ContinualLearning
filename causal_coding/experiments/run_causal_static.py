import os
import sys
import jax
import time
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.data import load_mnist, Task
from core.model import MLP
from core.runner import evaluate
from core.base import CausalState
from causal_coding.method import CausalMethod
from baselines.src.naive import NaiveMethod

key = jax.random.PRNGKey(0)
model = MLP([784, 256, 256, 10])
X, y, test_X, test_y = load_mnist()
task = Task(
    train_X=jnp.array(X),
    train_y=jnp.array(y),
    test_X=jnp.array(test_X),
    test_y=jnp.array(test_y),
    classes=list(range(10)),
)

# initiate params
params = model.init_params(key)
state = CausalState(
    old_params=params,
    influence_scores=jax.tree.map(lambda p: jnp.zeros_like(p), params),
)

method = CausalMethod(lr=0.01, batch_size=128, epochs=25, p=2.0, kappa=1e-8)
cc_start = time.time()
params, state, _ = method.train_task(model, params, state, task, task_idx=0)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
cc_end = time.time()
causal_acc = evaluate(model, params, task)

params = model.init_params(key)

naive = NaiveMethod(lr=0.01, batch_size=128, epochs=25)
naive_start = time.time()
params, _, _ = naive.train_task(model, params, None, task, task_idx=0)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
naive_end = time.time()

naive_acc = evaluate(model, params, task)


print(
    f"CausalMethod accuracy:{causal_acc * 100} % | time to train: {(cc_end - cc_start):.4f} sec"
)
print(
    f"NaiveMethod accuracy:{naive_acc * 100} % | time to train: {(naive_end - naive_start):.4f} sec"
)
