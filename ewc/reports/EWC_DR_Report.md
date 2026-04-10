# EWC Done Right (EWC-DR) Implementation and Results

**Framework:** JAX (from scratch)
**Dataset:** Split MNIST
**Evaluation Setting:** Class-Incremental Learning (single shared head, no task ID at inference)
**Reference Paper:** Liu & Chang, "Elastic Weight Consolidation Done Right for Continual Learning" (arXiv:2603.18596, 2026)

---

## 1. Background

After implementing vanilla EWC and observing its near-complete failure in Class-IL (final average accuracy ~20%, identical to naive training), we tested the Logits Reversal modification proposed in the EWC-DR paper.

The paper identifies a fundamental flaw in vanilla EWC: **gradient vanishing during Fisher Information computation**. When a well-trained model produces high-confidence correct predictions (`p_c` near 1.0), the gradient `(p_c - 1)` approaches zero, causing the Fisher values for the most important weights to be drastically underestimated. The result is that EWC's "rubber bands" become invisibly weak precisely on the weights that matter most.

---

## 2. The Modification

A single line change in the `compute_fisher` function. The log-likelihood is computed on the **negated** logits instead of the raw logits.

**Vanilla EWC:**
```
log_probs = log_softmax(logits)
```

**EWC-DR:**
```
log_probs = log_softmax(-logits)
```

Everything else stays the same — the training loss, the forward pass, the EWC penalty term. Only the importance estimation changes.

### Why It Works
Negating the logits inverts the softmax behavior. Instead of asking "how confident is the model in the correct prediction?" (which gives near-zero gradients when confidence is high), it asks "how would the gradient look if the model's confidence were inverted?" Weights that contributed most to the original confident prediction now produce the largest gradients in the reversed loss — properly reflecting their true importance.

The mathematical effect: for a confident correct prediction with `p_c = 0.99`:
- Vanilla EWC gradient term: `(p_c - 1)^2 = 0.0001` (vanishingly small)
- EWC-DR gradient term: `(1 - p_tilde_c)^2 ≈ 0.999` (large and meaningful)

---

## 3. Hyperparameter Adjustment

Because EWC-DR produces Fisher values orders of magnitude larger than vanilla EWC, lambda must be scaled down accordingly:

| Method | Best Lambda | Reason |
|--------|-------------|--------|
| Vanilla EWC | 5000 - 10000 | Compensating for tiny Fisher values |
| EWC-DR | 100 | Fisher values are now meaningful, smaller lambda needed |

Initial attempts with `lam=500` caused numerical explosion (`nan`) by Task 4, confirming that the much larger Fisher magnitudes require proportionally smaller lambda.

**Final config:** lr_task1=0.01, lr_ewc=0.001, 25 epochs, lam=100, num_fisher_samples=200, cumulative Fisher (Approach B).

---

## 4. Results

### Accuracy Matrix (EWC-DR, Class-IL)

|            | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------------|--------|--------|--------|--------|--------|
| After T1   | 99.9%  | 0.0%   | 0.0%   | 0.0%   | 0.0%   |
| After T2   | 85.3%  | 96.8%  | 0.0%   | 0.0%   | 0.0%   |
| After T3   | 46.4%  | 10.9%  | 98.6%  | 0.0%   | 0.0%   |
| After T4   | 55.1%  | 2.1%   | 16.8%  | 98.9%  | 0.0%   |
| After T5   | 43.8%  | 0.6%   | 0.05%  | 13.4%  | 96.1%  |

**Final Average Accuracy: 30.8%**

### Comparison Against Vanilla EWC and Naive Baseline

| Method | Final Avg Accuracy | T1 after T5 | Pattern |
|--------|-------------------|-------------|---------|
| Naive SGD | 19.7% | 0.0% | Total, immediate forgetting |
| Vanilla EWC (best: lam=10000) | 20.3% | 0.0% | Total, gradual forgetting |
| **EWC-DR (lam=100)** | **30.8%** | **43.8%** | **Partial retention** |

### Improvement Over Vanilla EWC

| Metric | Vanilla EWC | EWC-DR | Improvement |
|--------|-------------|--------|-------------|
| Final average accuracy | 20.3% | 30.8% | **+10.5%** |
| Task 1 retention after Task 5 | 0.0% | 43.8% | **+43.8%** |
| Task 1 retention after Task 2 | 8.0% | 85.3% | **+77.3%** |

---

## 5. Observations

### 5.1 What Works
- **No `nan` explosion** at moderate lambda (100). Vanilla EWC's most aggressive stable setting was lam=5000; EWC-DR is stable at much lower lambda values, which is itself an indicator of healthier Fisher estimates.
- **Strong early retention.** Task 1 holds at 85.3% after Task 2 — vanilla EWC managed only 8.0%.
- **Meaningful protection of the first task across the entire sequence.** Task 1 ends at 43.8%, which is dramatically better than the 0.0% from vanilla EWC.

### 5.2 What Still Fails
- **Middle tasks collapse.** Tasks 2 and 3 retention degrades to 0.6% and 0.05% by the end. The cumulative Fisher with moving anchor still struggles to protect tasks beyond the first.
- **Overall Class-IL forgetting persists.** While EWC-DR substantially improves over vanilla EWC, it does not solve catastrophic forgetting in Class-IL — only partially mitigates it.

### 5.3 Comparison to Paper's Claims
The EWC-DR paper reports approximately a 30 percentage point improvement on CIFAR-100 (from ~18% to ~47% final accuracy) using ResNet-18. Our setup uses Split MNIST with an MLP, but the relative improvement is comparable in spirit — the modification consistently provides a substantial boost over vanilla EWC. The paper's exact numbers cannot be directly compared because of different datasets, architectures, and training protocols.

---

## 6. Conclusion

A single-line modification — negating the logits during Fisher Information computation — transforms EWC from a method that fails completely in Class-IL (~0% retention on old tasks) to one that retains meaningful knowledge of the first task (~44%). This confirms the paper's central thesis: **vanilla EWC's failure is not a fundamental limitation of weight regularization, but a numerical artifact of how the Fisher Information Matrix is computed when models are well-trained.**

EWC-DR remains insufficient for full Class-IL continual learning — middle tasks still collapse and the output head bias problem is not fully addressed — but it represents a significant improvement at minimal cost. For the team's quarter plan, this provides a stronger regularization baseline against which Bayesian Causal Coding can be compared.

---

## 7. Implementation Note

The change required was approximately one character (adding `-` before `logits`). All other components — the network, training loop, EWC penalty, cumulative Fisher accumulation, evaluation pipeline — remained identical to the vanilla EWC implementation. This makes EWC-DR an extremely cheap upgrade for any existing EWC-based system.
