# Chapter 07: Picture Weights and Feature Learning

## Overview
This chapter explores how neural network weights can be visualized as 2D spatial pictures and how networks transition from direct template matching to multi-layer feature detection on the MNIST dataset.

## Core Concepts
* **Picture Weights:** Reshaping weight parameter vectors $\mathbf{w}_c \in \mathbb{R}^{784}$ back into $28 \times 28$ image matrices.
* **Up-weights vs. Down-weights:** Positive weights amplify matching strokes; negative weights penalize pixels that should be blank.
* **Dot Product as Template Matching:** Prediction via spatial cross-correlation and cosine alignment.
* **Limits of Single-Layer Vision:** Why variations in handwriting slant, thickness, and spatial overlap require intermediate hidden representation layers.

## Notebook
* **Notebook:** [07_picture_weights.ipynb](07_picture_weights.ipynb)
* **Status:** Completed & Executed
