# Continual Learning: An Experimental Study of Causal Coding and Bayesian Approaches

This repository contains an **experimental investigation** into continual learning, focusing on how different learning paradigms behave under sequential, non-stationary task settings.

## Experiment Scope

We systematically evaluate and compare three models under controlled conditions:

- **Elastic Weight Consolidation (EWC)** - stability-based baseline
- **Causal Coding (CC)** - causal modular learning with gated updates
- **Bayesian Causal Coding (BCC)** - uncertainty-aware Bayesian extension of CC

All models share comparable architectures and parameter counts to ensure fair comparison.

## Experimental Focus

The study examines whether continual learning can move beyond forgetting mitigation toward **cumulative knowledge integration**, specifically:

- Resistance to catastrophic forgetting
- Forward Transfer (learning new tasks faster)
- Backward Transfer (improving past tasks)
- Behavior under increasing task sequences

## Setup

- Architecture: small feed-forward neural networks
- Benchmarks: Split-MNIST, Permuted-MNIST
- Task regimes: 5 -> 50 sequential tasks
- Metrics: Accuracy (ACC), Backward Transfer (BWT), Forward Transfer (FWT)

## Objective

To empirically test how well causal structure and uncertainty modeling contribute to robust continual learning, and to assess how closely these methods approximate ideal modular learning assumptions in practice.

## Status

Ongoing experimentation - implementations, evaluations, and scaling analyses in progress.
