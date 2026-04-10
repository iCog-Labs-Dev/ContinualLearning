# EWC Implementation and Evaluation Report

**Framework:** JAX (implemented from scratch, no external ML libraries)
**Dataset:** Split MNIST (Class-Incremental Learning)
**Architecture:** MLP 784 - 512 - 512 - 10

---

## 1. Objective

Implement Elastic Weight Consolidation (EWC) as a baseline continual learning method for the team's quarter plan (Phase 2, Weeks 3-5). Evaluate its effectiveness in preventing catastrophic forgetting under the Class-Incremental Learning (CIL) setting, as specified in the Experimentation Protocol.

---

## 2. Background

### Catastrophic Forgetting
When a neural network is trained sequentially on multiple tasks, it overwrites previously learned knowledge. All information is stored in shared weights, and gradient descent has no mechanism to protect important old weights.

### Elastic Weight Consolidation (EWC)
EWC (Kirkpatrick et al., 2017) addresses this by adding a penalty term to the loss function that resists changing weights that were important for previous tasks:

```
L_EWC = L_new_task + (lambda/2) * SUM_i [ F_i * (theta_i - theta_old_i)^2 ]
```

Where:
- `F_i` = diagonal Fisher Information (importance of weight i for old tasks)
- `theta_old_i` = weight value after old task training
- `lambda` = hyperparameter controlling protection strength

### Class-Incremental Learning (CIL)
The hardest continual learning setting. A single shared output head classifies across all classes. No task identity is provided at inference time. The model must distinguish between all classes seen so far without knowing which task an input belongs to.

---

## 3. Implementation

### What Was Built (All From Scratch in JAX)
- **Neural network:** Forward pass, He initialization, ReLU activations
- **Training pipeline:** Cross-entropy loss, gradient descent, batched training
- **Fisher Information computation:** Per-example gradients via `jax.grad` + `jax.vmap`, squared and averaged
- **EWC loss function:** Standard task loss + Fisher-weighted quadratic penalty using `jax.tree.map`
- **EWC training loop:** Modified training step differentiating through the combined EWC loss
- **Evaluation:** Class-IL (raw argmax over all 10 classes) and Task-IL (masked to task-specific classes)
- **Data pipeline:** MNIST loading, normalization, splitting into 5 binary tasks

### EWC Variant: Cumulative Fisher (Approach B)
After each task, the new Fisher diagonal is added to a running cumulative Fisher. The anchor point (`old_params`) is updated to the current parameters. This provides accumulated protection for all previous tasks through a single penalty term.

### Task Configuration
| Task | Classes | Train Size | Test Size |
|------|---------|-----------|-----------|
| 1 | 0, 1 | ~12,665 | ~2,115 |
| 2 | 2, 3 | ~12,089 | ~2,042 |
| 3 | 4, 5 | ~11,263 | ~1,874 |
| 4 | 6, 7 | ~12,183 | ~1,986 |
| 5 | 8, 9 | ~11,800 | ~1,983 |

---

## 4. Baseline MLP Hyperparameter Tuning

Before EWC experiments, the base MLP was tuned on full MNIST to find the best configuration.

| Architecture | LR | Epochs | Test Accuracy |
|-------------|-----|--------|---------------|
| 64, 64 | 0.001 | 5 | 79.8% |
| 64, 64 | 0.001 | 15 | 88.9% |
| 64, 64 | 0.01 | 15 | 94.4% |
| 128, 128 | 0.01 | 15 | 94.8% |
| 256, 256 | 0.01 | 15 | 95.2% |
| 512, 512 | 0.01 | 15 | 95.3% |
| 512, 512 | 0.001 | 15 | 89.9% |
| **512, 512** | **0.01** | **25** | **96.3%** |

**Selected config:** 784-512-512-10, lr=0.01, 25 epochs, batch size 128.

---

## 5. Catastrophic Forgetting Demonstration

### Class-IL Results (Naive Sequential Training)

|            | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------------|--------|--------|--------|--------|--------|
| After T1   | 99.9%  | 0.0%   | 0.0%   | 0.0%   | 0.0%   |
| After T2   | 0.0%   | 98.5%  | 0.0%   | 0.0%   | 0.0%   |
| After T3   | 0.0%   | 0.0%   | 99.6%  | 0.0%   | 0.0%   |
| After T4   | 0.0%   | 0.0%   | 0.0%   | 99.6%  | 0.0%   |
| After T5   | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 98.3%  |

**Final average accuracy: 19.7% (only Task 5 retained)**

Catastrophic forgetting is complete and immediate. Every previous task drops to 0.0% accuracy upon learning a new task.

### Task-IL Results (Masked Evaluation)

When the model is told which 2 classes to choose from (Task-IL), accuracy on old tasks remains ~95-99% throughout sequential training. This reveals that internal feature representations are largely preserved — the forgetting is primarily in the output head, where logits for old classes become suppressed relative to new classes.

---

## 6. EWC Experiments

### 6.1 Local Experiments — Lambda Tuning

All experiments: lr_task1=0.01, lr_ewc=0.001, 25 epochs, Fisher samples=200

| Lambda | T1 after T2 | T1 after T5 | T5 after T5 | Stable? |
|--------|-------------|-------------|-------------|---------|
| 50 | 0.0% | 0.0% | 96.9% | Yes |
| 100 | 0.1% | 0.0% | 96.9% | Yes |
| 200 | 0.4% | 0.0% | 97.0% | Yes |
| 400 | 1.6% | 0.0% | 97.0% | Yes |
| 1000 | 3.0% | 0.0% | 96.9% | Yes |
| 5000 | 8.0% | 0.0% | 96.8% | Yes |
| 50000 | 31.0% | 46.3% | 0.0% | No (nan at Task 3) |

**Pattern:** Higher lambda increases short-term retention but all tasks eventually collapse. Beyond lambda=5000, cumulative Fisher causes numerical instability (nan/inf).

### 6.2 Google Colab Experiments (Tesla T4 GPU)

To test whether better Fisher estimates would improve results, experiments were run on Google Colab with a Tesla T4 GPU, allowing larger Fisher sample sizes. The `compute_fisher` function was modified to use batched gradient computation to avoid GPU out-of-memory errors.

| Lambda | Fisher Samples | T1 after T2 | T1 after T5 | Avg Acc (T5) |
|--------|---------------|-------------|-------------|--------------|
| 5000 | 200 | 26.7% | 0.0% | 19.9% |
| 1000 | 1000 | 25.8% | 0.0% | 19.5% |
| 500 | 2000 | 21.5% | 0.0% | 19.4% |
| 500 | 3000 | 21.3% | 0.0% | 19.4% |
| 10000 | 4000 | **36.0%** | 0.0% | **20.3%** |

**Key finding:** Increasing Fisher samples from 200 to 4000 did not improve results. Lambda remains the dominant factor. The best configuration (lam=10000, fisher=4000) achieved 36% Task 1 retention after Task 2, but still collapsed to 0% by Task 5.

### 6.3 Best EWC Result — Full Accuracy Matrix (lam=10000, fisher=4000)

|            | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------------|--------|--------|--------|--------|--------|
| After T1   | 99.9%  | 0.0%   | 0.0%   | 0.0%   | 0.0%   |
| After T2   | 36.0%  | 96.6%  | 0.0%   | 0.0%   | 0.0%   |
| After T3   | 17.3%  | 2.6%   | 98.6%  | 0.0%   | 0.0%   |
| After T4   | 0.8%   | 0.0%   | 5.8%   | 99.3%  | 0.0%   |
| After T5   | 0.0%   | 0.0%   | 0.0%   | 5.0%   | 96.3%  |

EWC shows a gradual decay rather than the instant collapse of naive training, but the end result is the same — only the most recent task is retained.

---

## 7. Analysis: Why EWC Fails in Class-IL

### 7.1 Output Head Bias (Primary Cause)
When training on a new task (e.g., digits 4, 5), the output neurons for old classes (0, 1, 2, 3) receive no learning signal. Their logits drift and become systematically smaller than the logits for recent classes. Since Class-IL uses argmax over all 10 classes, the model always predicts a recent class — regardless of whether internal features can still distinguish old classes.

The Task-IL experiments confirm this: when forced to choose only between a task's 2 classes, accuracy on old tasks remains ~99%. The features are preserved; the output layer is broken.

### 7.2 Moving Anchor Problem
In cumulative Fisher (Approach B), the anchor point `old_params` is updated after each task. The penalty pulls weights toward the most recent anchor, not toward each individual task's optimal point. Protection for early tasks erodes as the anchor shifts further from their optimal configuration.

### 7.3 Diagonal Fisher Approximation
The full Fisher Information Matrix captures correlations between weights (e.g., weights A and B are individually unimportant but their specific combination is critical). The diagonal approximation discards all correlation information, systematically underestimating the true importance of weight configurations.

### 7.4 Literature Confirmation
This result is consistent with published research. The team's Experimentation Protocol includes EWC specifically "to demonstrate the characteristic CIL failure mode." Regularization-only methods are known to be insufficient for Class-IL — replay-based methods (storing examples from old tasks) are generally required.

---

## 8. Results Summary

| Method | Final Avg Accuracy | Task 1 Retention | Forgetting |
|--------|-------------------|-----------------|------------|
| Naive SGD (no protection) | 19.7% | 0.0% | Total, immediate |
| EWC (best config: lam=10000) | 20.3% | 0.0% | Total, gradual |
| Theoretical upper bound (joint training) | ~96% | ~96% | None |

EWC provides marginal improvement over naive training (+0.6% average accuracy) but does not meaningfully prevent catastrophic forgetting in the Class-Incremental Learning setting.

---

## 9. Technical Notes

### Compute Environment
- **Local:** Windows 11, CPU (JAX), Python 3.12
- **Colab:** Google Colab, Tesla T4 GPU (CUDA), JAX with GPU acceleration
- **Fisher batching:** The `compute_fisher` function was modified for Colab to process gradients in batches of 100, avoiding GPU out-of-memory errors when using large sample sizes

### Numerical Stability
Lambda values above ~5000 with cumulative Fisher caused loss to explode to infinity/nan by Task 3. This is due to accumulated Fisher values growing large, causing the penalty gradient to overwhelm the task loss gradient. Lowering the EWC learning rate to 0.001 (vs 0.01 for Task 1) partially mitigated this but did not eliminate the upper bound on usable lambda.

---

## 10. Conclusion and Next Steps

EWC has been successfully implemented from scratch in JAX and thoroughly evaluated. The implementation is correct — the results match published literature and confirm EWC's known limitations in the Class-Incremental Learning setting.

### What EWC Demonstrates
- The concept of per-weight importance via Fisher Information is sound
- EWC slows forgetting (gradual decay vs instant collapse) but cannot prevent it in CIL
- The fundamental bottleneck is the output head, which EWC cannot adequately protect

### Implications for the Quarter Plan
- EWC serves its intended role as a baseline showing the regularization failure mode in CIL
- The team's proposed Bayesian Causal Coding (BCC) approach, which treats weights as probability distributions and uses causal structure to direct updates, may address EWC's limitations by providing more targeted protection
- Replay-based methods should be considered as an additional baseline for comparison
