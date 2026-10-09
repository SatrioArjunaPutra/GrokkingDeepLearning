# Chapter 10: Convolutional Neural Networks

## Overview
This chapter explores Convolutional Neural Networks (CNNs) from scratch using pure NumPy, explaining why dense networks struggle with spatial vision and how convolutions achieve translation invariance.

## Core Concepts
* **Weight Sharing & Spatial Locality:** Sliding compact $3 \times 3$ filters across images rather than allocating separate parameters for every pixel.
* **2D Discrete Convolution:** The mathematics of receptive field inner products, zero-padding, and stride.
* **Feature Detectors:** Hand-crafted vs. learned kernels (Sobel horizontal, Sobel vertical, ridge/corner detectors).
* **Max Pooling:** Spatial downsampling ($2 \times 2$, stride 2) that reduces dimensionality by $75\%$ while providing translational invariance.
* **Full CNN Pipeline:** Constructing a modular forward pass: `Conv2D` $\to$ `ReLU` $\to$ `MaxPool` $\to$ `Flatten` $\to$ `Dense` $\to$ `Softmax`.

## Notebook
* **Notebook:** [10_convolutional_neural_networks.ipynb](10_convolutional_neural_networks.ipynb)
* **Status:** Completed & Executed
