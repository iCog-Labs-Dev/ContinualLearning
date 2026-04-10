# Catastrophic Forgetting Demonstration Report
 
**Framework:** JAX (from scratch)  
**Dataset:** Split MNIST  
**Evaluation Setting:** Class-Incremental Learning (single shared head, no task ID at inference)

---

## Setup

- **Architecture:** 784 - 512 - 512 - 10
- **Task Split:** 5 tasks, 2 classes each (0-1, 2-3, 4-5, 6-7, 8-9)
- **Training:** 25 epochs per task, lr=0.01, batch size 128, vanilla SGD
- **Method:** Naive sequential fine-tuning (no forgetting prevention)

---

## Accuracy Matrix (Class-IL)

|            | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------------|--------|--------|--------|--------|--------|
| After T1   | 99.9%  |        |        |        |        |
| After T2   | 0.0%   | 98.5%  |        |        |        |
| After T3   | 0.0%   | 0.0%   | 99.6%  |        |        |
| After T4   | 0.0%   | 0.0%   | 0.0%   | 99.6%  |        |
| After T5   | 0.0%   | 0.0%   | 0.0%   | 0.0%   | 98.3%  |

---

## Results Summary

| Metric | Value |
|--------|-------|
| Final Average Accuracy (all 5 tasks) | **19.7%** |
| Accuracy on most recent task | 98.3% |
| Accuracy on all previous tasks | 0.0% |
| Forgetting | Total (100% drop on every prior task) |

---

## Conclusion

Naive sequential training causes complete catastrophic forgetting in the Class-IL setting. After each new task, all previously learned tasks drop to 0.0% accuracy. The network only retains knowledge of the most recently trained task. This establishes the lower bound that EWC must improve upon.
