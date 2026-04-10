# Baseline MLP Hyperparameter Tuning Report

**Author:** Samuel  
**Date:** 2026-04-05  
**Framework:** JAX (from scratch, no external ML libraries)  
**Dataset:** MNIST (60,000 train / 10,000 test)  
**Task:** Handwritten digit classification (10 classes)

---

## Objective

Identify the best MLP configuration for use as the baseline neural network in our EWC (Elastic Weight Consolidation) continual learning experiments. The final architecture must match the team's Experimentation Protocol specification: `784 - 512 - 512 - 10`.

---

## Model Architecture

- **Type:** Fully connected Multi-Layer Perceptron (MLP)
- **Activation:** ReLU on hidden layers, raw logits on output
- **Output:** 10-class classification via softmax cross-entropy
- **Weight Init:** He initialization (`W * sqrt(2 / fan_in)`), biases initialized to zero
- **Optimizer:** Vanilla SGD (no momentum)
- **Batch Size:** 128 (fixed across all tests)

---

## Experiment Matrix

| Test | Hidden Layers | Learning Rate | Epochs | Final Loss | Test Accuracy |
|------|--------------|---------------|--------|------------|---------------|
| 1    | 64, 64       | 0.001         | 5      | 1.052      | 79.8%         |
| 2    | 64, 64       | 0.001         | 15     | 0.444      | 88.9%         |
| 3    | 64, 64       | 0.01          | 15     | 0.192      | 94.4%         |
| 4    | 128, 128     | 0.01          | 15     | 0.178      | 94.8%         |
| 5    | 256, 256     | 0.01          | 15     | 0.165      | 95.2%         |
| 6    | 512, 512     | 0.01          | 15     | 0.160      | 95.3%         |
| 7    | 512, 512     | 0.001         | 15     | 0.394      | 89.9%         |
| 8    | 512, 512     | 0.01          | 25     | 0.115      | **96.3%**     |

---

## Key Findings

1. **Learning rate has the largest impact.** Increasing LR from 0.001 to 0.01 improved accuracy by ~5% across all architectures (Test 2 vs 3, Test 7 vs 6). A small network with the right LR (Test 3: 94.4%) outperforms a large network with a low LR (Test 7: 89.9%).

2. **Larger networks help, with diminishing returns.** At fixed LR=0.01 and 15 epochs, scaling hidden layers from 64 to 512 improved accuracy by less than 1% per step (94.4% -> 94.8% -> 95.2% -> 95.3%).

3. **More training epochs consistently improve results.** Test 8 (25 epochs) achieved the best overall accuracy at 96.3%, and the loss was still decreasing, suggesting further improvement is possible.

4. **Loss convergence correlates with LR.** At LR=0.001, the loss is still high after 15 epochs (0.394-0.444), indicating the model has not converged. LR=0.01 reaches much lower loss in the same number of epochs.

---

## Selected Configuration for EWC Experiments

| Parameter        | Value            |
|------------------|------------------|
| Architecture     | 784 - 512 - 512 - 10 |
| Learning Rate    | 0.01             |
| Epochs per Task  | 25               |
| Batch Size       | 128              |
| Optimizer        | SGD              |
| Activation       | ReLU             |
| Weight Init      | He initialization|

**Rationale:** This matches the Experimentation Protocol's specified architecture, achieves the highest accuracy (96.3%), and provides a strong baseline from which to measure catastrophic forgetting and EWC's effectiveness.

---

## Next Steps

- Split MNIST into sequential tasks (5 tasks, 2 classes each) per the CIL protocol
- Demonstrate catastrophic forgetting with naive sequential training (no protection)
- Implement EWC (Fisher Information computation + penalty term)
- Compare naive vs EWC using Average Accuracy, Forgetting Measure, and Backward Transfer metrics
