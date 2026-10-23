# Chapter 12: Recurrent Neural Networks (RNNs) — Neural Networks that Write Like Shakespeare

## Overview
This chapter introduces **Recurrent Neural Networks (RNNs)**, the foundational architecture for processing sequential data. We build an RNN from scratch in pure NumPy, exploring how hidden state allows neural networks to maintain memory across time steps and generate text character-by-character.

## Core Concepts
* **The Problem with Feedforward Networks for Sequences:** Why standard dense networks cannot naturally handle variable-length sequential input.
* **Recurrence and Hidden State:** How RNNs maintain a hidden state vector $\mathbf{h}_t$ that accumulates information across time steps.
* **Backpropagation Through Time (BPTT):** Unrolling the recurrent computation graph and applying the chain rule across multiple time steps.
* **Vanishing and Exploding Gradients in RNNs:** Why gradients either shrink to near-zero or blow up when propagated through many time steps, and basic mitigation strategies.
* **Character-Level Language Modeling:** Training an RNN to predict the next character in a sequence, enabling text generation.
* **Text Generation via Sampling:** Using the trained RNN to generate new text by sampling from the output probability distribution at each time step.
* **Pure NumPy Implementation:** Building an RNN forward pass, BPTT, and text generation pipeline without any frameworks.

## Notebook
* **Notebook:** [12_recurrent_neural_networks.ipynb](12_recurrent_neural_networks.ipynb)
* **Status:** Completed & Executed
