# EWC in JAX — Continual Learning Baseline

Implementation of Elastic Weight Consolidation (EWC) and EWC Done Right (EWC-DR) from scratch in JAX for the Continual Learning team's quarter plan.

## Overview

This project evaluates EWC as a baseline continual learning method under the Class-Incremental Learning (CIL) setting on Split MNIST. Three methods are implemented and compared:

- **Naive SGD** — Sequential training with no forgetting prevention
- **EWC** — Vanilla Elastic Weight Consolidation (Kirkpatrick et al., 2017)
- **EWC-DR** — EWC with Logits Reversal (Liu & Chang, 2026)

## Results Summary

| Method | Final Avg Accuracy | Task 1 Retention |
|--------|-------------------|-----------------|
| Naive SGD | 19.7% | 0.0% |
| EWC (lam=1000) | 19.4% | 0.0% |
| EWC-DR (lam=100) | **30.8%** | **43.8%** |

Full experiment reports are available in `reports/`.

## Project Structure

```
ewc_jax/
├── README.md
├── requirements.txt
├── .gitignore
├── reports/                     # Experiment reports
│   ├── Baseline_MLP_Report.md
│   ├── Catastrophic_Forgetting_Report.md
│   ├── EWC_Final_Report.md
│   └── EWC_DR_Report.md
├── plots/                       # Generated heatmaps
├── notebooks/                   # Colab experiments
│   └── EWC_Colab_Experiment_Updated.ipynb
├── src/                         # Source code
│   ├── __init__.py
│   ├── model.py                 # MLP neural network
│   ├── data.py                  # MNIST loading and task splitting
│   ├── utils.py                 # Losses, metrics, plotting helpers
│   ├── naive.py                 # Naive sequential baseline
│   ├── ewc.py                   # Vanilla EWC
│   └── ewc_dr.py                # EWC Done Right (Logits Reversal)
└── experiments/                 # Runnable experiment scripts
    ├── run_naive.py
    ├── run_ewc.py
    └── run_ewc_dr.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

From the project root (`ewc_jax/`):

```bash
python experiments/run_naive.py
python experiments/run_ewc.py
python experiments/run_ewc_dr.py
```

## Configuration

Hyperparameters are set directly in each experiment script:

| Parameter | Naive | EWC | EWC-DR |
|-----------|-------|-----|--------|
| Architecture | 784-512-512-10 | 784-512-512-10 | 784-512-512-10 |
| LR (Task 1) | 0.01 | 0.01 | 0.01 |
| LR (EWC tasks) | — | 0.001 | 0.001 |
| Lambda | — | 1000 | 100 |
| Fisher samples | — | 200 | 200 |
| Epochs per task | 25 | 25 | 25 |
| Batch size | 128 | 128 | 128 |

## Key Findings

1. **Vanilla EWC fails in Class-IL** — performs no better than naive training (~20% avg accuracy). This is a known limitation confirmed by literature.
2. **EWC-DR significantly improves retention** — a single-line change (negating logits during Fisher computation) fixes the gradient vanishing problem, boosting average accuracy to 30.8%.
3. **The output head bias is the fundamental bottleneck** — Task-IL evaluation shows internal features are preserved (~99%), but the shared output head cannot maintain calibration across tasks.

## References

- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS, 2017)
- Liu & Chang, "Elastic Weight Consolidation Done Right for Continual Learning" (arXiv:2603.18596, 2026)

## Framework

Built entirely from scratch in JAX — no external ML frameworks (no Flax, no Optax, no PyTorch). Uses only `jax`, `jax.numpy`, and standard Python libraries.
