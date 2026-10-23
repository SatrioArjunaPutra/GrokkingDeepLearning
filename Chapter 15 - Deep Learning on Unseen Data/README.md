# Chapter 15: Deep Learning on Unseen Data — Introducing Federated Learning

## Overview
This chapter explores **Federated Learning**, a privacy-preserving paradigm that enables multiple parties to collaboratively train a shared machine learning model without sharing their raw data. We implement federated averaging and differential privacy mechanisms from scratch in pure NumPy.

## Core Concepts
* **The Privacy Problem in Machine Learning:** Why centralizing sensitive data (medical records, financial data, personal messages) for training creates unacceptable privacy risks.
* **Federated Learning Paradigm:** Training models across decentralized data holders while keeping raw data local — only model updates (gradients) are shared.
* **Federated Averaging (FedAvg):** The core algorithm where each client trains locally, sends weight updates to a central server, and the server averages them to produce the global model.
* **Homomorphic Encryption Intuition:** Understanding how encrypted computations can protect gradient information during transmission.
* **Differential Privacy:** Adding calibrated noise to gradients to provide formal mathematical guarantees that individual training examples cannot be reverse-engineered from model updates.
* **The Privacy-Accuracy Trade-off:** How stronger privacy guarantees (more noise) degrade model accuracy, and strategies for balancing them.
* **Pure NumPy Simulation:** Simulating a federated learning environment with multiple virtual clients, federated averaging, and differential privacy noise injection.

## Notebook
* **Notebook:** [15_federated_learning.ipynb](15_federated_learning.ipynb)
* **Status:** Completed & Executed
