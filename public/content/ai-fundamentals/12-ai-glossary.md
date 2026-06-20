---
title: "AI Glossary and Key Concepts"
topic: ai-fundamentals
order: 12
estimatedTime: "30 minutes"
difficulty: beginner
prerequisites: []
summary: "A comprehensive alphabetical glossary of essential AI and machine learning terms, complete with mathematical definitions and concise explanations."
---

## AI Glossary and Key Concepts

## Overview

This glossary provides concise definitions of the most important terms in artificial intelligence and machine learning. Terms are organized alphabetically. Mathematical formulas are included where they clarify the concept.

---

## A

### Activation Function

A non-linear function applied to a neuron's output to introduce non-linearity into the network. Common choices include ReLU, sigmoid, and tanh.

- **ReLU:** $f(x) = \max(0, x)$
- **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **Tanh:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

### Attention Mechanism

A technique that allows models to focus on relevant parts of the input when producing output. The scaled dot-product attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, $V$ are query, key, and value matrices, and $d_k$ is the key dimension.

## B

### Backpropagation

The algorithm for computing gradients of the loss function with respect to each weight in the network, using the chain rule:

$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_i}$$

This enables gradient descent to update weights efficiently across many layers.

### Batch Size

The number of training examples processed together before updating model weights. Larger batches provide more stable gradient estimates but require more memory.

## C

### Cross-Entropy Loss

A loss function commonly used for classification tasks. For a single sample with true class $y$ and predicted probabilities $\hat{y}$:

$$L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

For binary classification this simplifies to:

$$L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

### Convolution

A mathematical operation that slides a filter (kernel) over input data to extract local features. In 2D:

$$(\mathbf{I} * \mathbf{K})(i, j) = \sum_m \sum_n \mathbf{I}(i+m, j+n) \cdot \mathbf{K}(m, n)$$

## D

### Dropout

A regularization technique that randomly sets a fraction $p$ of neuron outputs to zero during training, preventing co-adaptation and reducing overfitting.

## E

### Embedding

A learned dense vector representation of discrete data (words, tokens, items) in a continuous vector space where similar items are close together.

### Epoch

One complete pass through the entire training dataset. Models typically train for many epochs until convergence.

## F

### Fine-Tuning

The process of taking a pre-trained model and continuing training on a smaller, task-specific dataset to adapt it to a new domain.

## G

### Gradient Descent

An optimization algorithm that iteratively updates parameters in the direction that reduces the loss:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

where $\eta$ is the learning rate and $\nabla_\theta L$ is the gradient of the loss with respect to parameters $\theta$.

### GAN (Generative Adversarial Network)

A framework with two networks: a generator $G$ that creates fake data and a discriminator $D$ that distinguishes real from fake. They train adversarially until $G$ produces realistic outputs.

## H

### Hyperparameter

A parameter set before training begins (not learned from data). Examples: learning rate, batch size, number of layers, dropout rate.

## I

### Inference

The process of using a trained model to make predictions on new, unseen data. Contrasted with training, inference does not update model weights.

## L

### Latent Space

A compressed, learned representation where high-dimensional data is encoded into lower-dimensional vectors. Autoencoders and VAEs map inputs to latent spaces.

### Learning Rate

The step size $\eta$ in gradient descent. Too large causes instability; too small causes slow convergence. Often scheduled to decay over training.

### Loss Function

A function that quantifies how far model predictions are from ground truth. Training minimizes this function. Examples: MSE, cross-entropy, hinge loss.

$$L_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

## N

### Normalization

Techniques that standardize inputs or intermediate representations to stabilize training. Batch normalization normalizes across the batch dimension:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

## O

### Overfitting

When a model memorizes training data rather than learning generalizable patterns. Characterized by low training loss but high validation loss. Combated with regularization, dropout, and data augmentation.

## P

### Parameter

A value learned during training (weights and biases). A model with millions of parameters has millions of learnable values.

## R

### Regularization

Techniques that prevent overfitting by constraining model complexity. L2 regularization adds a penalty proportional to squared weight magnitudes:

$$L_{\text{regularized}} = L_{\text{original}} + \lambda \sum_i w_i^2$$

where $\lambda$ controls the regularization strength.

## S

### Softmax

A function that converts a vector of real numbers into a probability distribution:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

All outputs are positive and sum to 1, making it suitable for multi-class classification.

### Stochastic Gradient Descent (SGD)

A variant of gradient descent that computes gradients on random subsets (mini-batches) of data rather than the full dataset, trading precision for computational efficiency.

## T

### Tensor

A multi-dimensional array that generalizes scalars (0D), vectors (1D), and matrices (2D) to arbitrary dimensions. The fundamental data structure in deep learning frameworks.

### Tokenization

The process of splitting text into smaller units (tokens) for processing by NLP models. Tokens may be words, subwords, or characters depending on the tokenizer.

### Transfer Learning

Using a model pre-trained on a large general dataset as a starting point for a new task. The pre-trained weights provide useful feature representations that can be fine-tuned with less data.

### Transformer

An architecture based entirely on attention mechanisms (no recurrence). The dominant architecture for NLP and increasingly for vision and other domains.

## V

### Vanishing Gradient

A problem in deep networks where gradients become extremely small in early layers during backpropagation, effectively halting learning. Addressed by architectures like ResNets (skip connections) and activation functions like ReLU.

## W

### Weight

A learnable parameter in a neural network that determines the strength of connection between neurons. Updated during training via backpropagation and gradient descent.

---

## Quick Reference Table

| Term | Category | Key Formula |
|------|----------|-------------|
| Softmax | Activation | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| Sigmoid | Activation | $\frac{1}{1+e^{-x}}$ |
| Cross-Entropy | Loss | $-\sum y_i \log \hat{y}_i$ |
| MSE | Loss | $\frac{1}{n}\sum(y-\hat{y})^2$ |
| Gradient Descent | Optimization | $\theta - \eta \nabla L$ |
| L2 Regularization | Regularization | $L + \lambda\sum w_i^2$ |
