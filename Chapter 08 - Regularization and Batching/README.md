# Chapter 08: Regularization and Batching

## Overview
This chapter addresses the challenge of overfitting in high-capacity deep neural networks by implementing Inverted Dropout regularization and Mini-Batch Gradient Descent from scratch.

## Core Concepts
* **The Overfitting Dilemma:** Memorization vs. generalization—when training loss converges to zero while test loss diverges.
* **Dropout Regularization:** Randomly muting hidden activations ($p = 0.5$) forces neurons to develop independent, non-co-adapted features.
* **Inverted Dropout:** Multiplying surviving activations by $2.0$ ($1 / (1 - p)$) during training to maintain activation magnitude expectations without altering test-time inference.
* **Mini-Batching:** Slicing dataset into blocks ($B = 100$) to leverage SIMD vectorization and compute variance-reduced gradient approximations.

## Notebook
* **Notebook:** [08_regularization_batching.ipynb](08_regularization_batching.ipynb)
* **Status:** Completed & Executed
