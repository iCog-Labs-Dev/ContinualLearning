import sys
import os
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.model import MLP
from core.data import load_mnist, split_into_tasks
from core.metrics import average_accuracy, backward_transfer, plot_accuracy_matrix
from core.base import EWCVanillaState
from core.config import get_config
from core.runner import run_experiment
from src.ewc_dr import EWCDRMethod

X, y, test_X, test_y = load_mnist()
# Load hyperparameters from YAML or fallback to defaults
config = get_config(
    default_method_kwargs=dict(lr=0.001, lr_task1=0.01, batch_size=128, epochs=25, lam=100, num_samples=200)
)
tasks = split_into_tasks(X, y, test_X, test_y, config.task.class_pairs)

# Initialize model using config dimensions
model = MLP([config.model.input_dim] + config.model.hidden_dims + [config.model.output_dim])
key = jax.random.PRNGKey(0)
params = model.init_params(key)

# Inject kwargs directly into the method
method = EWCDRMethod(**config.method_kwargs)
state = EWCVanillaState(anchors=[])

params, _, class_il_matrix, task_il_matrix = run_experiment(
    method, model, params, state, tasks
)

print(f"\nAverage Class-IL Accuracy: {average_accuracy(class_il_matrix) * 100:.2f}%")
print(f"Average Task-IL Accuracy: {average_accuracy(task_il_matrix) * 100:.2f}%")
print(f"Backward Transfer (Class-IL): {backward_transfer(class_il_matrix) * 100:.2f}%")
