# Chapter 14: Learning to Write Like Shakespeare — Long Short-Term Memory (LSTM)

## Overview
This chapter introduces the **Long Short-Term Memory (LSTM)** architecture, which solves the vanishing gradient problem inherent in vanilla RNNs. Using the autograd framework built in Chapter 13, we implement an LSTM from scratch and train it to generate Shakespeare-like text character-by-character.

## Core Concepts
* **The Vanishing Gradient Problem Revisited:** Why vanilla RNNs struggle to learn long-range dependencies — gradients shrink exponentially as they propagate through many time steps.
* **The LSTM Cell Architecture:** Understanding the four gates that control information flow:
  * **Forget Gate ($\mathbf{f}_t$):** Decides what information to discard from the cell state.
  * **Input Gate ($\mathbf{i}_t$):** Decides which new information to store in the cell state.
  * **Candidate Values ($\tilde{\mathbf{C}}_t$):** Creates a vector of new candidate values to be added.
  * **Output Gate ($\mathbf{o}_t$):** Decides what parts of the cell state to output as the hidden state.
* **The Cell State as a "Conveyor Belt":** How the cell state $\mathbf{C}_t$ provides a highway for gradients to flow through time with minimal degradation.
* **Character-Level Shakespeare Generation:** Training the LSTM on Shakespeare's works to generate new text that mimics the Bard's style.
* **Embedding Layer for Characters:** Using learned embeddings instead of one-hot encoding for efficient character representation.
* **Training with the Custom Autograd Framework:** Leveraging the `Tensor` class from Chapter 13 for automatic gradient computation.

## Notebook
* **Notebook:** [14_lstm_shakespeare.ipynb](14_lstm_shakespeare.ipynb)
* **Status:** Completed & Executed
