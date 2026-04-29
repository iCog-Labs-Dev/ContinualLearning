# Task-IL vs Class-IL: A Comparative Analysis

**Dataset:** Split MNIST (5 tasks, 2 classes per task)
**Architecture:** MLP ([784, 512, 512, 10])
**Methods Evaluated:** Naive Baseline, EWC Done Right (EWC-DR), Synaptic Intelligence (SI)

---

## 1. Background Settings

In prior experiments, performance was evaluated purely in the **Class-Incremental Learning (Class-IL)** setting. In Class-IL, the model is asked to predict from all 10 classes without knowing which task the input comes from. 

We have now explicitly separated evaluation into two scenarios:
1. **Class-IL (Strict):** Output mask = `None`. The model must select from all 10 possible digits.
2. **Task-IL (Guided):** Output mask = `task.classes`. The model is told the task identity at inference time (e.g., "this image is either a 0 or a 1") and only the logits for those specific classes are considered.

This split reveals a crucial insight: are these methods failing to remember the *features*, or are they just failing to calibrate the *output heads*?

---

## 2. Results Summary

### Final Average Accuracies

| Method | Class-IL (Strict) | Task-IL (Guided) | Gap |
|--------|-------------------|------------------|-----|
| Naive Baseline | 19.66% | 98.30% | 78.64% |
| Synaptic Intelligence (SI) | 24.78% | 96.72% | 71.94% |
| EWC Done Right (EWC-DR) | 36.07% | 97.54% | 61.47% |

*Note: In Class-IL, random guessing yields 10%. In Task-IL, random guessing yields 50%. However, near-perfect retention (~98%) in Task-IL proves the internal features are intact.*

---

## 3. Detailed Method Breakdown

### 3.1 Naive Baseline (No Regularization)

In Naive SGD, the model overwrites previous weights. 
- **Class-IL:** Fails completely (19.66% final average). The output logit magnitudes for the most recent task (Task 5: classes 8 and 9) completely dominate all previous tasks. Tasks 1-4 evaluate to exactly 0.00%.
- **Task-IL:** Succeeds surprisingly well (98.30% final average). Even though the newer tasks push the weights around, the internal decision boundary between `0` and `1` (Task 1) remains largely intact. When forced to ignore the massive logits of `8` and `9`, the network can still accurately distinguish a `0` from a `1`.

### 3.2 Synaptic Intelligence (SI)

- **Class-IL:** Slight improvement over baseline (24.78%). It shows some resistance to catastrophic forgetting in the middle tasks (Task 1 retains 30.21%), but the recency bias still largely overrides older knowledge.
- **Task-IL:** High retention (96.72%). The penalty actively prevents drift in important parameters, preserving the task-specific sub-networks. Interestingly, this is slightly *lower* than Naive in Task-IL, indicating the SI penalty might be overly restrictive, slightly harming plasticity for subsequent tasks while only mildly helping Class-IL.

### 3.3 EWC Done Right (EWC-DR)

- **Class-IL:** Best performer by a wide margin (36.07%). EWC-DR successfully mitigates the vanishing gradient problem in Fisher estimation, allowing it to lock down crucial parameters. Task 1 finishes at an impressive 28.46% (and was at 52.86% right before Task 5). 
- **Task-IL:** High retention (97.54%). Like the others, when the output bias is neutralized by the task mask, the core feature representations demonstrate near-perfect stability.

---

## 4. Key Takeaways

1. **The "Forgetting" is Mostly in the Head, Not the Body:** 
   The massive divergence between Task-IL (>96% for all methods) and Class-IL (<37% for all methods) proves that the internal feature extractors (the hidden layers) are actually *not* catastrophically forgetting how to separate the distinct classes within a task. 

2. **Recency Bias Dominates:**
   The catastrophic failure in Class-IL is primarily a calibration issue. As the model trains on Task $T$, the biases and weights connected to the output nodes for classes in Task $T$ grow large. At inference without a mask, these massive logits simply overwhelm the logits of Tasks $1$ through $T-1$.

3. **Regularization Methods Attack the Wrong Problem:**
   Traditional parameter-isolation methods (like Vanilla EWC and SI) focus heavily on freezing the hidden representations. However, as the Naive Task-IL results show (98.30%), those representations aren't heavily degraded anyway. EWC-DR provides the best Class-IL boost because its modified Fisher logic happens to impose a stricter structural hold that somewhat restricts the output head drift, but it still falls victim to the fundamental recency bias.

**Conclusion:** Future efforts should pivot slightly away from pure weight-space regularization (which solves the already-solved Task-IL problem) and focus on mechanisms that calibrate the output layer across tasks, such as replay buffers, bias correction layers, or contrastive learning.