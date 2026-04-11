# EWC Stability Improvements — Experimental Report

**Framework:** JAX (from scratch)
**Dataset:** Split MNIST (5 tasks, 2 classes each)
**Evaluation Setting:** Class-Incremental Learning
**Architecture:** MLP 784 - 512 - 512 - 10
**Base Method:** EWC-DR (Logits Reversal) with Online decay

---

## 1. Motivation

EWC-DR with Logits Reversal produces much larger Fisher Information values than vanilla EWC, which properly reflects weight importance but creates numerical instability. Higher lambda values (needed for stronger protection) cause the penalty gradient to explode to `nan`. This limits the usable lambda range and caps EWC-DR's potential.

We investigated several mathematical approaches to extend the stable operating range of EWC-DR, allowing higher lambda values and potentially better forgetting prevention.

---

## 2. The Root Cause

The penalty gradient for each weight is:

```
penalty_grad_i = lambda * F_i * (theta_i - theta_old_i)
```

When all three terms are large simultaneously (high lambda, large Fisher from Logits Reversal, significant weight drift), the penalty gradient overwhelms the task gradient. Weight updates oscillate with growing amplitude, reaching infinity within a few training steps, then producing `nan` via `inf * 0`.

---

## 3. Approaches Tested

### 3.1 Fisher Normalization

**Method:** After computing Fisher for each task, normalize per-layer by dividing by the maximum value plus a small epsilon:

```
fisher = fisher / (max(fisher) + epsilon)
```

This bounds all Fisher values to the range [0, 1] while preserving the relative ranking of weight importance within each layer.

**Result:**

| Config | Avg Acc | T1 after T5 | Stable? |
|--------|---------|-------------|---------|
| EWC-DR, lam=200, no normalization | 36.7% | 58.1% | Yes |
| EWC-DR, lam=1000, no normalization | nan | — | No |
| **EWC-DR, lam=1000, normalized** | **43.0%** | **87.1%** | **Yes** |
| EWC-DR, lam=1000, normalized, no decay | 43.3% | 69.3% | Yes |
| EWC-DR, lam=1500, normalized | nan at T5 | — | No |

**Analysis:** Normalization extended the usable lambda range from ~200 to ~1000-1200. This enabled 5x stronger protection, improving average accuracy from 36.7% to 43.0% and Task 1 retention from 58.1% to 87.1%. The best overall result across all experiments.

Without decay, similar average accuracy (43.3%) but different retention pattern — protection distributed more evenly across tasks (T1: 69.3%, T4: 34.7%) vs concentrated on Task 1 with decay (T1: 87.1%, T4: 20.5%).

### 3.2 Log-Fisher Scaling

**Method:** Apply logarithmic compression to Fisher values:

```
fisher = log(1 + fisher)
```

Compresses the range sub-linearly — small values pass through nearly unchanged, large values are reduced. Preserves relative ranking.

**Result:**

| Config | Avg Acc | Stable? |
|--------|---------|---------|
| EWC-DR, lam=1000, log scaling | nan at T2 | No |

**Analysis:** Log compression is insufficient for EWC-DR's Fisher values. `log(1 + 500) = 6.2` is still much larger than normalization's `500/500 = 1.0`. At `lam=1000`, the penalty gradient is ~6x larger than with normalization, exceeding the stability threshold.

**Verdict:** Dropped. Normalization is strictly superior for this use case.

### 3.3 Exponential Moving Average (EMA) Anchor

**Method:** Instead of hard-switching the anchor point after each task, blend the old and new positions:

```
old_params = alpha * old_params + (1 - alpha) * params
```

With `alpha=0.5`, the anchor retains 50% memory of its previous position. Borrowed from target networks in reinforcement learning (DQN).

**Result:**

| Config | Avg Acc | T1 after T5 | Stable? |
|--------|---------|-------------|---------|
| EWC-DR + norm, lam=1000, alpha=0.0 (hard switch) | 43.0% | 87.1% | Yes |
| EWC-DR + norm, lam=1000, alpha=0.5 (EMA) | 19.1% | 1.5% | Yes |

**Analysis:** EMA dramatically degraded performance. The blended anchor converges to a weighted average of all previous task positions — a point in parameter space that is not optimal for any individual task. The penalty pulls weights toward this meaningless compromise, providing no useful protection.

The hard switch works better because the anchor is at least a position that was genuinely good for the most recent task. Combined with cumulative Fisher, this provides reasonable (if imperfect) protection.

**Verdict:** Dropped. Hard anchor switch is superior.

---

## 4. Full Comparison Table

All methods tested throughout the project, ordered by average accuracy:

| Method | Lambda | Decay | Norm | Anchor | Avg Acc | T1 after T5 |
|--------|--------|-------|------|--------|---------|-------------|
| Naive SGD | — | — | — | — | 19.7% | 0.0% |
| EMA Anchor (alpha=0.5) | 1000 | 0.9 | Yes | EMA | 19.1% | 1.5% |
| Vanilla EWC | 1000 | 1.0 | No | Hard | 19.4% | 0.0% |
| Online EWC | 10000 | 0.9 | No | Hard | 20.4% | 0.0% |
| EWC-DR | 100 | 1.0 | No | Hard | 30.8% | 43.8% |
| Online EWC-DR | 200 | 0.9 | No | Hard | 36.7% | 58.1% |
| **EWC-DR + Normalization** | **1000** | **1.0** | **Yes** | **Hard** | **43.3%** | **69.3%** |
| **Online EWC-DR + Norm** | **1000** | **0.9** | **Yes** | **Hard** | **43.0%** | **87.1%** |

---

## 5. Key Findings

### What Works
1. **Logits Reversal (EWC-DR)** is the single most impactful improvement, fixing the gradient vanishing problem in Fisher computation. Responsible for +11% average accuracy over vanilla EWC.
2. **Fisher Normalization** is the most effective stability technique, extending usable lambda by 5x and adding +6% average accuracy on top of EWC-DR.
3. **Online decay** provides marginal accuracy improvement but significantly changes the retention pattern — concentrating protection on the earliest tasks.

### What Doesn't Work
1. **Log-Fisher Scaling** provides insufficient compression for EWC-DR's large Fisher values.
2. **EMA Anchor** creates a meaningless compromise position that hurts all tasks.

### The Fundamental Limitation Remains
Even the best configuration (43.3%) is far from solving Class-IL forgetting. The output head bias — where logits for old classes become suppressed relative to new classes — cannot be addressed by any regularization-based importance weighting scheme. This confirms the literature's conclusion that replay-based or architectural methods are needed for true Class-IL continual learning.

---

## 6. Improvement Progression

```
Naive SGD          19.7%  ████████████████████
Vanilla EWC        19.4%  ███████████████████
Online EWC         20.4%  ████████████████████
EWC-DR             30.8%  ███████████████████████████████
Online EWC-DR      36.7%  █████████████████████████████████████
EWC-DR + Norm      43.3%  ███████████████████████████████████████████
Online EWC-DR+Norm 43.0%  ███████████████████████████████████████████
```

The largest jumps came from:
- Logits Reversal: +11.1%
- Fisher Normalization: +6.3%
- Online Decay: +5.9%
