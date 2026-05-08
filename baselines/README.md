# Continual Learning Baselines in JAX

Implementation of Elastic Weight Consolidation (EWC), EWC Done Right (EWC-DR), Synaptic Intelligence (SI), and various continual learning strategies from scratch in JAX for the Continual Learning team's quarter plan.

## Overview

This project evaluates EWC and other continual learning methods under the Class-Incremental Learning (CIL) and Task-Incremental Learning settings on Split MNIST. The methods implemented and compared include:

- **Naive SGD** — Sequential training with no forgetting prevention
- **EWC** — Vanilla Elastic Weight Consolidation (Kirkpatrick et al., 2017)
- **EWC-DR** — EWC with Logits Reversal (Liu & Chang, 2026)
- **Online EWC & Online EWC-DR** — Scalable EWC with a running EMA of Fisher Information Matrices
- **EWC with EMA** — Exponential Moving Average for tracking model weights for testing
- **Synaptic Intelligence (SI)** — Path integral-based parameter masking (Zenke et al., 2017)

## Results Summary

| Method | Final Avg Accuracy | Task 1 Retention |
|--------|-------------------|-----------------|
| Naive SGD | 19.7% | 0.0% |
| EWC (lam=1000) | 19.4% | 0.0% |
| EWC-DR (lam=100) | **36.1%** | **28.5%** |

*Note: Class-IL accuracy metrics vary depending on exact settings. See the detailed reports below for full comparisons between Class-IL and Task-IL performance.*

## Experiment Reports

Full experiment and analysis reports have been migrated to Google Docs:

- [Baseline MLP Hyperparameter Tuning Report](https://docs.google.com/document/d/1CaRuEW9mpMueLoOl1xlQAs5GnsXSjPHgP_iDikKWgt0/edit?usp=sharing)
- [Catastrophic Forgetting Demonstration Report](https://docs.google.com/document/d/18S5AXcB8VF09Ex0_87OluSuUXz1Ib-WGdFFqdXwioHU/edit?usp=sharing)
- [EWC Implementation and Evaluation Report](https://docs.google.com/document/d/1hDkrHsU0JGpW7Hge-PXZa5LrERt_KMJAIWPME1yAKUg/edit?usp=sharing)
- [EWC Done Right (EWC-DR) Implementation and Results](https://docs.google.com/document/d/17EbLuT7gOoDbF9nXtc0M8odrLIw8g0oTnwY-BS0V6bY/edit?usp=sharing)
- [EWC Stability Improvements — Experimental Report](https://docs.google.com/document/d/1CIsjc1LQVu_P9RuuKhUZ7VMYHXiw63wJx9LNz4acdzI/edit?usp=sharing)
- [Online EWC and EWC-DR Variants — Experiment Report](https://docs.google.com/document/d/13FpCt4lepfR9fm6WmTPI0OAF1vHAPLxFk70wAeg1CaU/edit?usp=sharing)
- [Synaptic Intelligence (SI) — Implementation and Evaluation Report](https://docs.google.com/document/d/1bjMCOO00ndN99X_aBiZultrnyKNVU0kvkcVjHiSFLZY/edit?usp=sharing)
- [Task-IL vs Class-IL: A Comparative Analysis](https://docs.google.com/document/d/1G4tlfo3IEy38hbiTihRjqsM50jdScIIGnJ71dO4DTjs/edit?usp=sharing)

## Project Structure

```
baselines/
├── README.md
├── plots/                       # Generated heatmaps
├── notebooks/                   # Colab experiments
│   └── EWC_Colab_Experiment.ipynb
├── src/                         # Baseline method implementations
│   ├── __init__.py
│   ├── naive.py                 # Naive sequential baseline
│   ├── ewc.py                   # Vanilla EWC
│   ├── ewc_dr.py                # EWC Done Right (Logits Reversal)
│   └── si.py                    # Synaptic Intelligence
└── experiments/                 # Runnable experiment scripts
    ├── run_naive.py
    ├── run_ewc.py
    ├── run_ewc_dr.py
    ├── run_online_ewc.py
    ├── run_online_ewc_dr.py
    ├── run_si.py
    └── run_ewc_with_ema.py
```

The MLP architecture, MNIST data pipeline, evaluation metrics, and
experiment runner that all baselines share live in `core/` at the
project root — not inside this folder. The experiment scripts add the
project root to `sys.path` automatically so no manual configuration is
needed.

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

From inside the `baselines/` folder:

```bash
python experiments/run_naive.py
python experiments/run_ewc.py
python experiments/run_ewc_dr.py
# Run other experiments accordingly
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
2. **EWC-DR significantly improves retention** — a single-line change (negating logits during Fisher computation) fixes the gradient vanishing problem, boosting average accuracy significantly.
3. **The output head bias is the fundamental bottleneck** — Task-IL evaluation shows internal features are preserved (~98-99%), but the shared output head cannot maintain calibration across tasks due to recency bias.

## References

- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS, 2017)
- Liu & Chang, "Elastic Weight Consolidation Done Right for Continual Learning" (arXiv:2603.18596, 2026)
- Zenke et al., "Continual Learning Through Synaptic Intelligence" (ICML, 2017)

## Framework

Built entirely from scratch in JAX — no external ML frameworks (no Flax, no Optax, no PyTorch). Uses only `jax`, `jax.numpy`, and standard Python libraries.
