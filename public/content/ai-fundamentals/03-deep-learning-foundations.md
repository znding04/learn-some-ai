---
title: "Deep Learning Foundations"
level: intermediate
topic: ai-fundamentals
order: 3
---

# Deep Learning Foundations

## Overview

Deep learning is the subfield of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data. It's the technology behind modern image recognition, language models, and much more.

### Neural Network Architecture

A neural network is organized into **layers**:

1. **Input layer** — Receives raw data (pixels, words, numbers)
2. **Hidden layers** — Process and transform representations. "Deep" means many hidden layers.
3. **Output layer** — Produces the final prediction

Each layer contains **neurons** (nodes). Each neuron computes:

$$z = \sum_{i=1}^{n} w_i x_i + b$$
$$a = \sigma(z)$$

Where $w_i$ are weights, $b$ is a bias, $x_i$ are inputs, and $\sigma$ is an activation function. The weights are the learnable parameters — they determine what the network "knows."

### Activation Functions

Activation functions introduce non-linearity, allowing networks to learn complex patterns. Without them, stacking layers would just produce a linear transformation.

**ReLU (Rectified Linear Unit)** — The default choice for hidden layers:

$$\text{ReLU}(z) = \max(0, z)$$

Simple, fast to compute, and avoids the vanishing gradient problem for positive values.

**Sigmoid** — Squashes output to $(0, 1)$, useful for binary classification:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Tanh** — Squashes output to $(-1, 1)$, zero-centered:

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Softmax** — Used in the output layer for multi-class classification. Converts a vector of scores into a probability distribution:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

### Backpropagation

Training a neural network means finding weights that minimize the loss function. Backpropagation computes the gradient of the loss with respect to every weight in the network.

The intuition: start from the output, compute how much each weight contributed to the error, and propagate that information backward through the layers.

Mathematically, backpropagation is just the **chain rule** applied repeatedly:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_3} \cdot \frac{\partial a_3}{\partial z_3} \cdot \frac{\partial z_3}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_1}$$

Each factor is a local gradient — easy to compute at each layer. The full gradient is just their product.

### The Chain Rule in Action

Consider a simple 2-layer network predicting a single value. Forward pass:

$$z_1 = w_1 x + b_1, \quad a_1 = \text{ReLU}(z_1)$$
$$z_2 = w_2 a_1 + b_2, \quad \hat{y} = z_2$$
$$L = (\hat{y} - y)^2$$

Backward pass (computing $\frac{\partial L}{\partial w_1}$):

$$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$$
$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat{y}} \cdot a_1$$
$$\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial \hat{y}} \cdot w_2$$
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_1} \cdot \mathbb{1}[z_1 > 0] \cdot x$$

### Mini-Batch Gradient Descent

Computing the gradient over the entire dataset (batch gradient descent) is expensive. Computing it for one sample (stochastic gradient descent, SGD) is noisy. The practical compromise is **mini-batch gradient descent**:

- Split the training data into small batches (typically 32–256 samples)
- Compute the gradient on each batch
- Update weights after each batch

This gives a good balance between computation speed and gradient stability. One full pass through the dataset is called an **epoch**.

## Key Concepts

- **Neuron**: The basic unit — computes a weighted sum plus bias, then applies an activation
- **Activation Function**: Non-linear function (ReLU, sigmoid, etc.) enabling complex representations
- **Backpropagation**: Algorithm to compute gradients using the chain rule, layer by layer
- **Chain Rule**: The mathematical foundation of backpropagation
- **Mini-Batch**: A subset of training data used for one gradient update
- **Epoch**: One complete pass through the training dataset

## Code Examples

```python
import torch
import torch.nn as nn

# Define a simple 3-layer neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Create model, loss function, and optimizer
model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop (one mini-batch)
X_batch = torch.randn(32, 10)   # 32 samples, 10 features
y_batch = torch.randn(32, 1)

# Forward pass
predictions = model(X_batch)
loss = criterion(predictions, y_batch)

# Backward pass (backpropagation)
optimizer.zero_grad()
loss.backward()        # Computes all gradients
optimizer.step()       # Updates weights
print(f"Loss: {loss.item():.4f}")
```

## Diagrams

```
Neural Network Architecture:

Input Layer    Hidden Layer 1    Hidden Layer 2    Output
  (x₁) ─────┐
              ├──→ [h₁] ─────┐
  (x₂) ─────┤                ├──→ [h₃] ──┐
              ├──→ [h₂] ─────┤            ├──→ (ŷ)
  (x₃) ─────┘                └──→ [h₄] ──┘

Each arrow = weight (w)
Each node = σ(Σ wᵢxᵢ + b)
```

```
Activation Functions:

ReLU:          Sigmoid:         Tanh:
 y│    /       y│   ___         y│   ___
  │   /         │  /   1         │  /   1
  │  /          │ /              │ /
──┼──── x    ──┼──── x       ──┼──── x
  │             │\              │\
  │             │ \___ 0         │ \___ -1
```

## Exercises

1. **Manual backprop**: For $f(x) = (2x + 3)^2$, compute $\frac{df}{dx}$ using the chain rule. Then verify with a numerical gradient.
2. **Code challenge**: Modify the PyTorch code to add a second hidden layer with 16 neurons. How does the number of parameters change?
3. **Experiment**: Train the network for 100 epochs on random data. Plot the loss curve. What do you observe?

## Further Reading

- 3Blue1Brown: "Neural Networks" series on YouTube
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning* (Chapter 6: Deep Feedforward Networks)
- PyTorch tutorials: https://pytorch.org/tutorials/
