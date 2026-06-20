import sys
import os
import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.base import EWCVanillaState
from core.config import get_config
from benchmarker import CLBenchmark
from src.ewc_dr import EWCDRMethod

X, y, test_X, test_y = load_mnist()

# Load hyperparameters from YAML or fallback to defaults
config = get_config(
    default_method_kwargs=dict(lr=0.001, lr_task1=0.01, batch_size=128, epochs=25, lam=100, num_samples=200)
)

# print(f"Loaded config: {config.model_dump()}")

tasks = split_into_tasks(X, y, test_X, test_y, config.task.class_pairs)

# Initialize model using config dimensions
model = MLP([config.model.input_dim] + config.model.hidden_dims + [config.model.output_dim])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

# Inject kwargs directly into the method
method = EWCDRMethod(**config.method_kwargs)
state = EWCVanillaState(anchors=[])

CLBenchmark(method=method, model=model, tasks=tasks, name="ewc_dr").run(params, state)
