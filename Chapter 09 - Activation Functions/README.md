# Chapter 09: Activation Functions

## Overview
This chapter explores non-linear activation functions (Sigmoid, Tanh, ReLU) and how neural networks output calibrated categorical probability distributions using Softmax and Cross-Entropy loss.

## Core Concepts
* **Non-Linear Activations:** Breaking linear subspace collapse so multi-layer architectures can approximate arbitrary non-linear functions.
* **Vanishing Gradient:** Why squashing functions (Sigmoid with max derivative $0.25$, Tanh with max derivative $1.0$) saturate at extreme values and impede deep backpropagation.
* **Softmax Distribution:** Exponentiating and normalizing unnormalized logits into calibrated probabilities where $\sum \hat{y}_k = 1.0$.
* **Softmax + Cross-Entropy Synergy:** The gradient of categorical Cross-Entropy with Softmax reduces elegantly to $\boldsymbol{\delta} = \hat{\mathbf{y}} - \mathbf{y}$.

## Notebook
* **Notebook:** [09_activation_functions.ipynb](09_activation_functions.ipynb)
* **Status:** Completed & Executed
