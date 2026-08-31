"""
Run the Discriminative PCN on Split-MNIST using the Task-IL protocol only.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import jax

from core.data import load_mnist, split_into_tasks
from benchmarker import CLBenchmark
from benchmarker.results import BenchmarkResults
from benchmarker.logger import save_results
from benchmarker.plotter import plot_all

from discrimunative.src.model_discriminative import DiscriminativePCN
from discrimunative.src.method_discriminative import PCNMethodDiscriminative
from discrimunative.src.pcn_wrapper_discriminative import PCNDiscriminativeModelWrapper


class TaskILOnlyBenchmark(CLBenchmark):
    """
    Custom benchmark runner that only executes the Task-IL protocol,
    skipping the Class-IL runs to save training time.
    """
    def run(self, params, state) -> BenchmarkResults:
        # 1. Compute baselines (Class-IL and Task-IL)
        (
            class_il_baselines,
            task_il_baselines,
            class_il_nll_baselines,
            task_il_nll_baselines,
        ) = self._evaluator.compute_baselines(self.model, params, self.tasks)

        # 2. Run Task-IL protocol only
        task_il_matrix, task_il_nll_matrix, task_il_bce_matrix = self._run_protocol(
            params, state, protocol_name="Task-IL", task_il_training=True
        )
        self.method.task_il_training = False

        # Create dummy matrices/metrics for Class-IL so downstream metrics calculation doesn't crash
        T = len(self.tasks)
        dummy_matrix = [[0.0] * T for _ in range(T)]
        dummy_baselines = [0.0] * T

        metrics = self._compute_metrics(
            dummy_matrix,
            task_il_matrix,
            dummy_matrix,
            task_il_nll_matrix,
            task_il_bce_matrix,
            dummy_baselines,
            task_il_baselines,
            dummy_baselines,
            task_il_nll_baselines,
        )

        results = BenchmarkResults(
            method_name=self.name,
            config=self.config,
            class_il_matrix=dummy_matrix,
            task_il_matrix=task_il_matrix,
            metrics=metrics,
            class_il_baselines=dummy_baselines,
            task_il_baselines=task_il_baselines,
        )

        results.summary()
        save_results(
            self.name,
            metrics,
            dummy_matrix,
            task_il_matrix,
            self.config,
            class_il_nll_matrix=dummy_matrix,
            task_il_nll_matrix=task_il_nll_matrix,
            task_il_bce_matrix=task_il_bce_matrix,
        )
        plot_all(self.name, dummy_matrix, task_il_matrix)

        return results


# Config (same as run_pcn_benchmark.py)
LAYER_SIZES     = [784, 512, 512, 10]
SEED            = 0
BATCH_SIZE      = 128
EPOCHS          = 35
ETA_X           = 0.1
ETA_W           = 0.005
INFERENCE_STEPS = 50
CLASS_PAIRS     = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

print("PCN Benchmark — Discriminative (Task-IL Only)")
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

# Run Task-IL only benchmark
TaskILOnlyBenchmark(
    method=method,
    model=model,
    tasks=tasks,
    name="pcn_discriminative_task_il"
).run(params, None)
