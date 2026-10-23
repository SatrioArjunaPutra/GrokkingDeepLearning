# Chapter 11: Neural Networks that Understand Language (Word Embeddings)

## Overview
This chapter introduces **Word Embeddings** — dense vector representations that allow neural networks to process and understand natural language. We explore how words can be mapped into continuous vector spaces where semantic relationships are preserved as geometric relationships.

## Core Concepts
* **The Limitations of One-Hot Encoding:** Why sparse, orthogonal vectors fail to capture semantic similarity between words.
* **Word Embeddings as Learned Representations:** Training dense, low-dimensional vectors that encode semantic meaning through context.
* **Word Analogy & Vector Arithmetic:** The famous `king - man + woman ≈ queen` relationship emerges naturally from embedding geometry.
* **Filling in the Blanks (CBOW):** Predicting a target word from its surrounding context words.
* **Skip-Gram Architecture:** Predicting context words from a single target word — the inverse of CBOW.
* **Embedding Similarity:** Using cosine similarity and dot products to measure semantic closeness in the embedding space.
* **Pure NumPy Implementation:** Building a word embedding training pipeline from scratch without any deep learning frameworks.

## Notebook
* **Notebook:** [11_word_embeddings.ipynb](11_word_embeddings.ipynb)
* **Status:** Completed & Executed
