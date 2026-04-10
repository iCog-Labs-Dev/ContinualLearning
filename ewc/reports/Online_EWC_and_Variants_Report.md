# Online EWC and EWC-DR Variants — Experiment Report

**Framework:** JAX (from scratch)
**Dataset:** Split MNIST (5 tasks, 2 classes each)
**Evaluation Setting:** Class-Incremental Learning
**Architecture:** MLP 784 - 512 - 512 - 10

---

## 1. Methods Evaluated

### Online EWC
Modification to vanilla EWC that applies a decay factor to the cumulative Fisher Information before adding each new task's Fisher. Prevents unbounded growth of the penalty term.

```
cumulative_fisher = decay * cumulative_fisher + new_fisher
```

### EWC-DR (Logits Reversal)
Modification to vanilla EWC that negates logits before computing Fisher Information, fixing the gradient vanishing problem that causes vanilla EWC to underestimate weight importance.

### Online EWC-DR (Combined)
Combines both improvements: decay for stable accumulation + Logits Reversal for accurate importance estimation.

---

## 2. Results

### Full Comparison Table

| Method | Lambda | Decay | Samples | Epochs | Avg Acc | T1 after T5 |
|--------|--------|-------|---------|--------|---------|-------------|
| Naive SGD | — | — | — | 25 | 19.7% | 0.0% |
| Vanilla EWC | 1000 | 1.0 | 200 | 25 | 19.4% | 0.0% |
| Online EWC | 1000 | 0.9 | 200 | 25 | 19.4% | 0.0% |
| Online EWC | 10000 | 0.9 | 300 | 25 | 20.4% | 0.0% |
| EWC-DR | 100 | 1.0 | 200 | 25 | 30.8% | 43.8% |
| **Online EWC-DR** | **200** | **0.9** | **200** | **25** | **36.7%** | **58.1%** |
| Online EWC-DR | 150 | 0.9 | 200 | 100 | 29.9% | 29.5% |

### Best Result — Online EWC-DR Accuracy Matrix (lam=200, decay=0.9, 25 epochs)

|            | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------------|--------|--------|--------|--------|--------|
| After T1   | 99.9%  | 0.0%   | 0.0%   | 0.0%   | 0.0%   |
| After T2   | 89.5%  | 96.5%  | 0.0%   | 0.0%   | 0.0%   |
| After T3   | 52.0%  | 19.3%  | 98.4%  | 0.0%   | 0.0%   |
| After T4   | 62.6%  | 7.5%   | 22.5%  | 98.7%  | 0.0%   |
| After T5   | 58.1%  | 3.8%   | 0.4%   | 25.7%  | 95.7%  |

---

## 3. Key Findings

### Online EWC Alone Has Minimal Impact
Adding decay to vanilla EWC (without Logits Reversal) showed negligible improvement. With decay=0.9 and lam=1000, results were identical to vanilla EWC. Even with lam=10000 (enabled by decay preventing nan), average accuracy only reached 20.4%. The core problem remains: vanilla Fisher values are too small due to gradient vanishing.

### EWC-DR Is the Critical Improvement
Logits Reversal is responsible for the majority of the performance gain. EWC-DR without decay (30.8%) already massively outperforms Online EWC with decay (20.4%).

### Combining Both Gives the Best Result
Online EWC-DR (decay + Logits Reversal) achieved the highest average accuracy (36.7%) and best Task 1 retention (58.1%). The decay allows slightly higher lambda (200 vs 100) without numerical instability, providing stronger protection.

### More Epochs Can Hurt
Increasing from 25 to 100 epochs degraded Online EWC-DR performance from 36.7% to 29.9%. Longer training drives higher model confidence, which even with Logits Reversal begins to saturate the Fisher estimates. The sweet spot is training long enough for good task performance but not so long that importance estimation degrades.

### Numerical Stability Remains a Challenge
Higher lambda values (300-500) with EWC-DR + decay still caused nan explosions by Task 4-5. The reversed logits produce much larger Fisher values, and even with decay=0.9, accumulation across many tasks can overflow. The usable lambda range for Online EWC-DR is approximately 100-250.

---

## 4. Conclusion

The combination of Online EWC's decay mechanism with EWC-DR's Logits Reversal provides the strongest regularization-based result in our experiments. However, even the best configuration (36.7% average accuracy) remains far from solving Class-IL forgetting. The fundamental limitation of regularization-only methods — inability to maintain output head calibration across tasks — persists regardless of how accurately weight importance is estimated.

### Improvement Hierarchy

```
Naive (19.7%) → Vanilla EWC (19.4%) → Online EWC (20.4%) → EWC-DR (30.8%) → Online EWC-DR (36.7%)
```

The largest single improvement came from Logits Reversal (+11.1%), not from Online decay (+0.7%).
