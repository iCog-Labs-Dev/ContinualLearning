"""
Run the PCN on Split-MNIST using the scalable forward_xl evaluation.

Difference vs run_pcn_benchmark.py:
  - Model evaluation uses forward_xl (1 inference pass, free top layer)
    instead of the exhaustive forward (10 passes, one per class).
  - Everything else (training, data, hyperparams) is identical.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import jax

from core.data import load_mnist, split_into_tasks
from benchmarker import CLBenchmark
from src.model import GenerativePCN
from src.method import PCNMethod
from src.pcn_wrapper import PCNModelWrapperXL


# Config (same as run_pcn_benchmark.py)
LAYER_SIZES     = [784, 512, 512, 10]
SEED            = 0
BATCH_SIZE      = 128
EPOCHS          = 35
ETA_X           = 0.1
ETA_W           = 0.005
INFERENCE_STEPS = 50
CLASS_PAIRS     = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

print("PCN Benchmark — forward_xl (single-pass top-layer readout)")
print(f"  layer_sizes={LAYER_SIZES}  inference_steps={INFERENCE_STEPS}")

# Data
X, y, test_X, test_y = load_mnist()
tasks = split_into_tasks(X, y, test_X, test_y, CLASS_PAIRS)

# Model — use PCNModelWrapperXL so evaluation calls forward_xl
pcn    = GenerativePCN(LAYER_SIZES)
params = pcn.init_params(jax.random.PRNGKey(SEED))
model  = PCNModelWrapperXL(pcn)

method = PCNMethod(
    layer_sizes=LAYER_SIZES,
    eta_x=ETA_X,
    eta_w=ETA_W,
    inference_steps=INFERENCE_STEPS,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    seed=SEED,
)

# Run
CLBenchmark(method=method, model=model, tasks=tasks, name="pcn_xl").run(params, None)
