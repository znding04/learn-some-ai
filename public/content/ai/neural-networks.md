# Neural Networks: From Perceptrons to Backpropagation

## Introduction

Neural networks are function approximators inspired loosely by biological neurons. They form the foundation of deep learning and modern AI. This lesson builds from the simplest model (perceptron) to the core learning algorithm (backpropagation) that makes neural networks powerful.

## The Perceptron

A perceptron is the simplest neural network unit — takes inputs, computes a weighted sum, and outputs a binary decision.

### Structure
$$z = \vec{w} \cdot \vec{x} + b$$
$$y = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

### Geometric Interpretation
The perceptron divides the input space with a hyperplane (line in 2D, plane in 3D). Points on one side output 1; points on the other output 0.

### Limitation
Perceptrons can only learn **linearly separable** patterns. The classic XOR problem cannot be solved by a single perceptron — this limitation drove the development of multi-layer networks.

## Multi-Layer Perceptrons (MLPs)

Stack multiple layers of neurons to learn non-linear patterns:

| Layer | Role |
|-------|------|
| Input layer | Features $\vec{x}$ |
| Hidden layers | Learn intermediate representations |
| Output layer | Final prediction $\hat{y}$ |

### Forward Propagation
For layer $l$:
$$\vec{z}^{(l)} = W^{(l)} \vec{a}^{(l-1)} + \vec{b}^{(l)}$$
$$\vec{a}^{(l)} = f(\vec{z}^{(l)})$$

where $f$ is the activation function and $\vec{a}^{(0)} = \vec{x}$.

## Activation Functions

Non-linearities that allow networks to learn complex patterns:

### Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- Range: (0, 1)
- Historically popular, but suffers from vanishing gradients

### Tanh
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- Range: (-1, 1)
- Zero-centered, often outperforms sigmoid in hidden layers

### ReLU (Rectified Linear Unit)
$$f(x) = \max(0, x)$$
- Most common choice for modern networks
-Computationally efficient, helps with vanishing gradients

### Softmax
For multi-class classification (output layer):
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
- Outputs valid probability distribution (sums to 1)

## Loss Functions

The loss function measures how wrong the prediction is:

| Task | Loss Function |
|------|--------------|
| Binary classification | Binary cross-entropy |
| Multi-class classification | Categorical cross-entropy |
| Regression | Mean squared error (MSE) |

For classification with softmax outputs, we use **cross-entropy**:
$$L = -\sum_c y_c \log(\hat{y}_c)$$

This is the negative log-likelihood — minimizing it is equivalent to maximizing the probability of the correct class.

## Backpropagation

Backpropagation is the algorithm for computing gradients of the loss with respect to weights — it uses the chain rule from calculus.

### The Chain Rule
If $y = f(g(x))$, then:
$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

For a network: $\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial \vec{z}^{(l)}} \cdot \frac{\partial \vec{z}^{(l)}}{\partial W^{(l)}}$

### Backprop Algorithm
1. **Forward pass:** Compute all activations and the loss
2. **Backward pass:** Compute $\frac{\partial L}{\partial \vec{z}^{(L)}}$ at output layer
3. **Propagate:** Use chain rule to compute gradients layer by layer back to input
4. **Update:** $W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$

The gradients "flow backward" from loss to weights, hence "backpropagation."

## Gradient Descent

Update weights in the direction that reduces loss:

$$W^{(l)} := W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$

where $\alpha$ is the **learning rate** — the most important hyperparameter.

| Variant | Update rule |
|---------|------------|
| SGD | Update per sample |
| Mini-batch | Update per batch of samples |
| Adam | Adaptive learning rates per parameter |

## Practice Problems

1. **Why can't a single perceptron learn XOR?**
   <details><summary>Answer</summary>XOR is not linearly separable — there is no straight line that separates the (0,1) and (1,0) input classes from (0,0) and (1,1). You need at least 2 layers (1 hidden layer) to carve the feature space into non-linear regions.</details>

2. **Compute the gradient of ReLU: $f(x) = \max(0, x)$**
   <details><summary>Answer</summary>$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$ (in practice, often set to 0 at x=0)</details>

3. **In a 3-layer network, given $\frac{\partial L}{\partial \vec{z}^{(3)}}$ (output layer gradients), how do you compute $\frac{\partial L}{\partial W^{(2)}}$?**
   <details><summary>Answer</summary>Using chain rule: $\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial \vec{z}^{(3)}} \cdot \frac{\partial \vec{z}^{(3)}}{\partial W^{(2)}} = \frac{\partial L}{\partial \vec{z}^{(3)}} \cdot (\vec{a}^{(2)})^T$. The gradient w.r.t. layer-2 weights is the outer product of output-layer delta and layer-2 activations.</details>

## Key Takeaways

- Perceptrons compute $\vec{w} \cdot \vec{x} + b$ and output a binary decision — can only learn linearly separable patterns
- MLPs stack layers with non-linear activation functions to learn any continuous function
- Activation functions (ReLU, sigmoid, tanh) introduce the non-linearity that makes deep networks powerful
- Cross-entropy loss is standard for classification; MSE for regression
- Backpropagation applies the chain rule to compute gradients efficiently layer by layer
- Gradient descent updates weights opposite to the gradient direction, minimizing the loss
- The learning rate $\alpha$ controls how large each update step is — too large = unstable, too small = slow learning
