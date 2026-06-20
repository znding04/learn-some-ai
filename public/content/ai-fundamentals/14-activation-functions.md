---
title: "Activation Functions Deep Dive"
difficulty: intermediate
topic: ai-fundamentals
order: 14
estimatedTime: "30 minutes"
summary: "A deep dive into activation functions including sigmoid, tanh, ReLU, and modern alternatives like GELU and Swish, covering their mathematical properties, the vanishing gradient problem, and weight initialization."
---

## Activation Functions Deep Dive

## Overview

Activation functions introduce nonlinearity into neural networks, enabling them to learn complex patterns beyond simple linear relationships. Without activations, stacking layers would be equivalent to a single linear transformation — no matter how deep the network. Choosing the right activation function affects training speed, gradient flow, and model expressiveness. This lesson examines the most widely used activations, their mathematical properties, and the practical problems they solve (or create).

## Why Nonlinearity Matters

A linear function composed with another linear function is still linear:

$$f(g(\mathbf{x})) = \mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \mathbf{W}_2\mathbf{W}_1 \mathbf{x} + \mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2$$

This collapses to $\mathbf{W}'\mathbf{x} + \mathbf{b}'$ — a single-layer network. Nonlinear activations break this collapse, allowing deep networks to represent hierarchical features.

## The Classic Activations

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Output range**: $(0, 1)$

**Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

The sigmoid was the default activation for decades. It maps any real number to a probability-like output between 0 and 1, making it natural for binary classification output layers.

**Problems**:
- **Vanishing gradients**: The maximum derivative is $\sigma'(0) = 0.25$. In a deep network, multiplying many values less than 1 causes gradients to shrink exponentially toward zero, making early layers nearly impossible to train.
- **Not zero-centered**: Outputs are always positive, which can cause zig-zag dynamics in gradient updates.
- **Saturation**: For large $|z|$, the gradient is near zero — the neuron "saturates" and stops learning.

### Tanh (Hyperbolic Tangent)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

**Output range**: $(-1, 1)$

**Derivative**: $\tanh'(z) = 1 - \tanh^2(z)$

Tanh is zero-centered (solving one sigmoid issue) and has a steeper gradient around zero (maximum derivative of 1.0 vs 0.25). However, it still suffers from saturation and vanishing gradients in deep networks. It remains common in RNNs and LSTMs where bounded outputs help control information flow.

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

**Output range**: $[0, \infty)$

**Derivative**: $\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$

ReLU revolutionized deep learning (Glorot et al., 2011). Its advantages:
- **No saturation** (for positive inputs): gradient is always 1
- **Computationally cheap**: just a comparison and assignment
- **Sparsity**: many neurons output exactly 0, creating sparse representations
- **Fast convergence**: networks train 6x faster than sigmoid equivalents (Krizhevsky, 2012)

**Problems**:
- **Dying ReLU**: If a neuron's input is always negative (e.g., due to a large negative bias), its gradient is always 0 and it never updates. Once dead, it stays dead. In practice, 10-40% of neurons in a network can die.
- **Not zero-centered**: Outputs are always non-negative.

## Modern Activation Functions

### Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}$$

where $\alpha$ is a small constant (typically 0.01). This prevents dying neurons by allowing a small gradient when the input is negative. **Parametric ReLU (PReLU)** makes $\alpha$ a learnable parameter.

### ELU (Exponential Linear Unit)

$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

ELU produces negative outputs, making it zero-centered. It smoothly saturates to $-\alpha$ for large negative inputs, adding noise robustness. The exponential computation is slightly more expensive than ReLU.

### SELU (Scaled Exponential Linear Unit)

$$\text{SELU}(z) = \lambda \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

where $\lambda \approx 1.0507$ and $\alpha \approx 1.6733$ are mathematically derived constants. SELU is **self-normalizing**: under certain conditions (LeCun initialization, fully connected layers), activations converge toward zero mean and unit variance automatically. This eliminates the need for batch normalization in some architectures.

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z) = z \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$$

GELU is the default activation in Transformers (BERT, GPT). It can be interpreted as a smooth, probabilistic version of ReLU — multiplying the input by the probability that a standard Gaussian random variable is less than $z$. Unlike ReLU, it is smooth everywhere and non-monotonic near zero.

### Swish / SiLU

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

Discovered through automated search (Ramachandran et al., 2017), Swish consistently outperforms ReLU on deep networks. It is smooth, non-monotonic, and unbounded above. It is used in EfficientNet and many modern architectures.

## The Vanishing Gradient Problem in Detail

Consider a network with $L$ layers, all using sigmoid activation. The gradient of the loss with respect to the first layer is:

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \prod_{l=1}^{L} \sigma'(z^{(l)}) W^{(l)}$$

Since $\sigma'(z) \leq 0.25$ everywhere, the product of $L$ such terms shrinks exponentially:

$$(0.25)^L \rightarrow 0 \text{ as } L \rightarrow \infty$$

For a 10-layer sigmoid network, gradients at the first layer are attenuated by a factor of at least $(0.25)^{10} \approx 10^{-6}$.

ReLU mitigates this because its gradient is either 0 or 1 — no multiplicative shrinking for active neurons.

## Weight Initialization: The Key Partner

Activation functions and weight initialization are deeply coupled. Poor initialization can cause activations to explode or collapse regardless of the activation choice.

- **Xavier/Glorot initialization** (for sigmoid/tanh): $W \sim \mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$
- **He/Kaiming initialization** (for ReLU): $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$
- **LeCun initialization** (for SELU): $W \sim \mathcal{N}(0, \frac{1}{n_{in}})$

The idea: keep the variance of activations roughly constant across layers at initialization, so gradients neither vanish nor explode during the first few training steps.

```python
import torch.nn as nn

# He initialization for ReLU networks
layer = nn.Linear(256, 128)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

# Xavier initialization for tanh networks
layer2 = nn.Linear(256, 128)
nn.init.xavier_normal_(layer2.weight)
```

## Practical Guidelines

| Activation | Best For | Avoid When |
|-----------|---------|------------|
| ReLU | Default for hidden layers in CNNs, MLPs | Very deep nets prone to dying neurons |
| Leaky ReLU | When dying ReLU is a problem | — |
| GELU | Transformer architectures | Latency-sensitive inference |
| Swish/SiLU | Modern CNNs (EfficientNet) | — |
| Sigmoid | Binary output layer | Hidden layers |
| Tanh | RNN/LSTM gates | Deep feedforward nets |
| Softmax | Multi-class output layer | Hidden layers |

## Key Concepts

- **Nonlinearity**: Required for deep networks to learn beyond linear functions
- **Vanishing gradients**: Sigmoid/tanh shrink gradients exponentially in deep networks
- **Dying ReLU**: Neurons with permanently negative inputs stop learning
- **GELU/Swish**: Smooth, non-monotonic activations dominate modern architectures
- **Self-normalization**: SELU maintains activation statistics without batch normalization
- **Initialization coupling**: Activation choice dictates the correct weight initialization scheme

## Further Reading

- [Glorot & Bengio — Understanding the difficulty of training deep feedforward neural networks (2010)](https://proceedings.mlr.press/v9/glorot10a.html)
- [He et al. — Delving Deep into Rectifiers (2015)](https://arxiv.org/abs/1502.01852)
- [Hendrycks & Gimpel — Gaussian Error Linear Units (GELUs) (2016)](https://arxiv.org/abs/1606.08415)
- [Ramachandran et al. — Searching for Activation Functions (2017)](https://arxiv.org/abs/1710.05941)
- [Klambauer et al. — Self-Normalizing Neural Networks (2017)](https://arxiv.org/abs/1706.02515)
