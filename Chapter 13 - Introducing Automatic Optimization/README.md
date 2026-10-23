# Chapter 13: Introducing Automatic Optimization — Building a Deep Learning Framework

## Overview
This chapter builds a **Deep Learning Framework from scratch** by implementing an automatic differentiation (autograd) engine in pure Python/NumPy. This framework enables us to define arbitrary computational graphs and automatically compute gradients, eliminating the need for manual backpropagation derivation.

## Core Concepts
* **The Motivation for Automatic Differentiation:** Why manually deriving and coding gradients for every new architecture is error-prone and unsustainable.
* **Tensors as the Core Abstraction:** Wrapping NumPy arrays into `Tensor` objects that track their computational history (creators and creation operations).
* **The Computational Graph:** How each operation builds a directed acyclic graph (DAG) that records the forward computation for later backward traversal.
* **Autograd via Reverse-Mode Differentiation:** Implementing `.backward()` to traverse the computational graph in reverse topological order, accumulating gradients via the chain rule.
* **Operator Overloading:** Implementing `__add__`, `__neg__`, `__sub__`, `__mul__`, and matrix multiplication to make tensor operations feel natural.
* **Building Layer Abstractions:** Creating reusable `Linear`, `Sequential`, `Sigmoid`, `Tanh`, `MSELoss`, and `CrossEntropyLoss` layers on top of the autograd engine.
* **SGD Optimizer:** Implementing stochastic gradient descent as a method that updates all trainable parameters.
* **End-to-End Training Loop:** Training a complete neural network on MNIST using only our custom framework.

## Notebook
* **Notebook:** [13_automatic_optimization.ipynb](13_automatic_optimization.ipynb)
* **Status:** Completed & Executed
