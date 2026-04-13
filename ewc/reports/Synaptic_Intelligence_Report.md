# Synaptic Intelligence (SI) — Implementation and Evaluation Report

**Framework:** JAX (from scratch)
**Dataset:** Split MNIST (5 tasks, 2 classes each)
**Evaluation Setting:** Class-Incremental Learning
**Architecture:** MLP 784 - 512 - 512 - 10

---

## 1. Background

Synaptic Intelligence (Zenke et al., 2017) is a regularization-based continual learning method that measures weight importance **during** training rather than after. While EWC computes a snapshot of importance via the Fisher Information Matrix at the end of each task, SI accumulates each weight's contribution to loss reduction along the entire optimization path.

### How SI Computes Importance

At each training step, for each weight:
```
contribution += -gradient * (theta_after_step - theta_before_step)
```

After all training on a task completes:
```
omega = max(0, contribution / (total_distance^2 + epsilon))
```

Where `total_distance = theta_final - theta_initial` for that task.

The penalty term is structurally identical to EWC:
```
L_SI = L_task + (lambda/2) * sum(omega_i * (theta_i - theta_old_i)^2)
```

### Key Differences from EWC

| Aspect | EWC | SI |
|--------|-----|-----|
| When measured | After training (snapshot) | During training (accumulated) |
| What it measures | Loss surface curvature | Actual contribution to learning |
| Extra computation | Separate Fisher pass over data | Nearly zero — tracked alongside training |
| Gradient vanishing | Yes (at convergence) | No — gradients are large during training |

---

## 2. Implementation

SI was implemented as a new class `SIMethod` in `src/si.py`, following the same stateless OOP pattern as `EWCMethod`:

- `_si_train_step`: Modified training step that tracks per-weight contribution sums alongside normal parameter updates
- `SIMethod.train_task`: Training loop that initializes contribution tracking, trains with penalty from previous tasks' omega, then computes new omega at task completion
- Omega normalization: Optional per-layer normalization (`omega / (max + epsilon)`) to enable higher lambda values

---

## 3. Hyperparameter Tuning

| Lambda | Normalization | Avg Acc | BWT | T1 after T2 | T1 after T5 | Stable? |
|--------|--------------|---------|-----|-------------|-------------|---------|
| 1 | No | 19.5% | -73.8% | 41.5% | 0.0% | Yes |
| 10 | No | 21.5% | -71.8% | 73.0% | 7.8% | Yes |
| 30 | No | 24.2% | -68.6% | 84.3% | 20.4% | Yes |
| 40 | No | 24.9% | -67.6% | 86.7% | 24.1% | Yes (limit) |
| 100 | No | nan at T3 | — | 92.2% | — | No |
| 500 | Yes | 24.6% | -65.4% | 95.4% | 28.8% | Yes |
| 1000 | Yes | nan | — | — | — | No |

### Observations

**Lambda sensitivity:** SI has a narrow usable lambda range (1-40) without normalization. Beyond 40, cumulative omega causes numerical explosion by Task 3.

**Normalization helps but less dramatically than for EWC-DR:** Omega normalization extended the usable range from ~40 to ~500, but the best normalized result (24.6%) only marginally improved over the best unnormalized result (24.9%).

**Strong early retention:** At `lam=40`, Task 1 retains 86.7% after Task 2 — comparable to EWC-DR. But this collapses rapidly over subsequent tasks.

---

## 4. Comparison with EWC Variants

| Method | Avg Acc | BWT | T1 after T5 |
|--------|---------|-----|-------------|
| Naive SGD | 19.7% | -99.4% | 0.0% |
| Vanilla EWC | 19.4% | -99.5% | 0.0% |
| **SI (best: lam=40)** | **24.9%** | **-67.6%** | **24.1%** |
| **SI + Norm (lam=500)** | **24.6%** | **-65.4%** | **28.8%** |
| EWC-DR | 30.8% | -73.0% | 43.8% |
| Online EWC-DR | 36.7% | -62.0% | 58.1% |
| EWC-DR + Norm | 43.3% | -55.0% | 69.3% |
| Online EWC-DR + Norm | 43.0% | -48.0% | 87.1% |

### Key Finding

SI outperforms vanilla EWC (+5% avg accuracy) because it does not suffer from gradient vanishing — importance is measured during training when gradients are large and meaningful. However, SI is significantly outperformed by EWC-DR variants (-18% avg accuracy vs best EWC-DR).

This suggests that **fixing how importance is computed** (Logits Reversal in EWC-DR) is more impactful than **changing when importance is computed** (during training in SI vs after training in EWC).

---

## 5. Shared Failure Pattern

SI exhibits the same numerical instability pattern observed across all regularization methods:

1. **Tasks 1-2:** Strong retention (85-95% on Task 1 after Task 2)
2. **Task 3 onwards:** Cumulative importance grows, penalty gradients become too large, either causing explosion (nan) or forcing lambda to be too low for effective protection
3. **Final state:** Only the most recent task is retained with meaningful accuracy

The cumulative penalty growth scales roughly linearly with the number of tasks, explaining why methods that work well for 2 tasks consistently degrade at 3-5 tasks.

---

## 6. Conclusion

SI is a valid alternative to EWC that eliminates the separate Fisher computation pass and avoids gradient vanishing at convergence. It provides modest improvement over vanilla EWC in the Class-IL setting. However, for this project's purposes, EWC-DR with Fisher Normalization remains the strongest regularization baseline at 43.3% average accuracy — nearly double SI's best result.
