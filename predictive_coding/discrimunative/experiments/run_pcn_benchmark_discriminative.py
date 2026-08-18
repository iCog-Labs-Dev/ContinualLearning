"""
Run the Discriminative PCN on Split-MNIST using the ContinualLearning benchmark.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import jax

from core.data import load_mnist, split_into_tasks
from benchmarker import CLBenchmark

from discrimunative.src.model_discriminative import DiscriminativePCN
from discrimunative.src.method_discriminative import PCNMethodDiscriminative
from discrimunative.src.pcn_wrapper_discriminative import PCNDiscriminativeModelWrapper

# Config (same as run_pcn_benchmark.py)
LAYER_SIZES     = [784, 512, 512, 10]
SEED            = 0
BATCH_SIZE      = 128
EPOCHS          = 35
ETA_X           = 0.1
ETA_W           = 0.005
INFERENCE_STEPS = 50
CLASS_PAIRS     = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

print("PCN Benchmark — Discriminative (single-pass top-layer readout)")
print(f"  layer_sizes={LAYER_SIZES}  inference_steps={INFERENCE_STEPS}")

# Data
X, y, test_X, test_y = load_mnist()
tasks = split_into_tasks(X, y, test_X, test_y, CLASS_PAIRS)

# Model
pcn    = DiscriminativePCN(LAYER_SIZES, activation="tanh")
params = pcn.init_params(jax.random.PRNGKey(SEED))
model  = PCNDiscriminativeModelWrapper(pcn)

method = PCNMethodDiscriminative(
    layer_sizes=LAYER_SIZES,
    eta_x=ETA_X,
    eta_w=ETA_W,
    inference_steps=INFERENCE_STEPS,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    seed=SEED,
)

# Run
CLBenchmark(method=method, model=model, tasks=tasks, name="pcn_discriminative").run(params, None)
