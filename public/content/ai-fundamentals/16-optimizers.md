---
title: "Optimizers and Learning Rates"
difficulty: intermediate
topic: ai-fundamentals
order: 16
estimatedTime: "30 minutes"
summary: "Walks through the evolution of optimization algorithms from vanilla SGD through momentum, AdaGrad, RMSprop, Adam, and AdamW, plus learning rate schedules like cosine annealing and warmup."
---

## Optimizers and Learning Rates

## Overview

Training a neural network means navigating a high-dimensional loss landscape to find good parameters. The optimizer is the algorithm that decides *how* to move through this landscape — not just which direction (the gradient tells us that), but how far, how fast, and how to avoid common traps like saddle points, sharp minima, and oscillation. This lesson walks through the evolution of optimization algorithms from vanilla SGD to Adam and beyond, and explores how learning rate schedules can dramatically improve training.

## Vanilla Stochastic Gradient Descent

The simplest optimizer computes the gradient on a mini-batch and takes a step:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta$ is the learning rate. SGD is elegant but has several well-known problems:

- **Oscillation in narrow valleys**: When the loss surface is much steeper in one direction than another (high condition number), SGD oscillates across the narrow dimension while making slow progress along the long dimension.
- **Sensitivity to learning rate**: Too high and it diverges; too low and it takes forever.
- **Saddle points**: In high dimensions, most critical points are saddle points (not minima). SGD can get stuck near them because the gradient is close to zero.

## SGD with Momentum

Momentum (Polyak, 1964) accelerates SGD by accumulating a velocity vector:

$$v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9). The velocity $v_t$ is an exponentially weighted moving average of past gradients. This has two key effects:

1. **Acceleration**: Consistent gradients in the same direction accumulate, speeding up movement through flat regions and past saddle points.
2. **Damping**: Oscillating gradients (positive then negative) cancel out, reducing the zig-zag behavior in narrow valleys.

Think of a ball rolling down a hill: momentum lets it barrel through small bumps and flat patches rather than stopping at every dip.

### Nesterov Accelerated Gradient (NAG)

Nesterov momentum (1983) is a clever modification: instead of computing the gradient at the current position, compute it at the *lookahead* position $\theta_t - \eta \beta v_{t-1}$:

$$v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t - \eta \beta v_{t-1})$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

This "look before you leap" approach provides a correction — if the momentum is carrying you too far, the gradient at the lookahead position pushes back. NAG converges faster than standard momentum on convex problems and often helps in practice.

```python
# PyTorch: SGD with Nesterov momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)
```

## Adaptive Learning Rate Methods

The key insight: different parameters may need different learning rates. Parameters associated with frequently occurring features should have smaller updates; parameters for rare features should have larger updates.

### AdaGrad (Adaptive Gradient)

AdaGrad (Duchi et al., 2011) maintains a per-parameter sum of squared gradients:

$$G_t = G_{t-1} + (\nabla_\theta \mathcal{L})^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_\theta \mathcal{L}$$

Parameters with large accumulated gradients get smaller effective learning rates. This is excellent for sparse data (NLP, recommender systems) but has a critical flaw: $G_t$ only grows, so the learning rate monotonically decreases and eventually becomes too small to learn.

### RMSprop (Root Mean Square Propagation)

RMSprop (Hinton, 2012, unpublished lecture notes) fixes AdaGrad's aggressive decay by using an exponentially weighted moving average instead of a sum:

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)(\nabla_\theta \mathcal{L})^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla_\theta \mathcal{L}$$

with $\beta$ typically 0.9. This "forgets" old gradients, maintaining a relevant estimate of recent gradient magnitudes.

### Adam (Adaptive Moment Estimation)

Adam (Kingma & Ba, 2015) combines momentum with RMSprop — it maintains both the first moment (mean) and second moment (uncentered variance) of the gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L})^2$$

Because $m_t$ and $v_t$ are initialized to zero, they are biased toward zero in early steps. Adam applies **bias correction**:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The update rule:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

Adam is the **default optimizer** for most deep learning tasks. It converges faster than SGD in practice and requires less learning rate tuning.

### AdamW (Adam with Decoupled Weight Decay)

Loshchilov & Hutter (2019) showed that L2 regularization in Adam doesn't work the same as weight decay — the adaptive learning rates interfere. AdamW fixes this by decoupling weight decay from the gradient-based update:

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

AdamW is the standard optimizer for training Transformers and large language models.

```python
# AdamW — the default for Transformers
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

## Learning Rate Schedules

The learning rate doesn't have to be constant. Schedules adjust $\eta$ during training, often providing significant improvements.

### Step Decay

Reduce the learning rate by a factor every $N$ epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / N \rfloor}$$

Common choice: halve every 30 epochs ($\gamma = 0.5, N = 30$).

### Cosine Annealing

Smoothly decay the learning rate following a cosine curve:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

Cosine annealing is the standard schedule for vision models and is increasingly used for LLM training. It starts with a high learning rate (for fast initial progress), smoothly reduces it (for fine-grained convergence), and avoids the abrupt drops of step decay.

### Warmup + Cosine Decay

For Transformers, the standard approach combines linear warmup with cosine decay:

1. **Warmup** (first $W$ steps): Linearly increase $\eta$ from 0 to $\eta_{\max}$
2. **Cosine decay** (remaining steps): Cosine anneal from $\eta_{\max}$ to $\eta_{\min}$

Warmup prevents early instability when the model hasn't yet learned meaningful representations and the adaptive estimates ($m_t, v_t$) are unreliable.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

warmup = LinearLR(optimizer, start_factor=0.01, total_iters=1000)
cosine = CosineAnnealingLR(optimizer, T_max=50000, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[1000])

for step in range(51000):
    loss = train_step(model, batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### One-Cycle Policy

Smith & Topin (2019) proposed training with a single cycle: ramp the learning rate up to a maximum, then bring it down. This super-convergence technique trains faster and often finds better solutions.

## Optimizer Comparison

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| SGD + Momentum | Best generalization, well-understood | Slow convergence, sensitive to LR | CNNs, when tuning budget allows |
| Adam | Fast convergence, low tuning | Worse generalization than SGD | Rapid prototyping, RNNs |
| AdamW | Correct weight decay for Transformers | Slightly more memory | Transformers, LLMs |
| RMSprop | Good for non-stationary problems | Outdated by Adam | RNNs (legacy) |

## Key Concepts

- **Momentum**: Accumulates velocity to accelerate through flat regions and damp oscillation
- **Adaptive learning rates**: Per-parameter rates based on gradient history (AdaGrad, RMSprop, Adam)
- **Bias correction**: Compensates for zero-initialization of moment estimates in Adam
- **AdamW**: Decoupled weight decay — the standard for Transformer training
- **Cosine annealing**: Smooth learning rate decay following a cosine curve
- **Warmup**: Gradual learning rate increase to stabilize early training
- **Generalization gap**: SGD often generalizes better than Adam, but takes longer

## Further Reading

- [Kingma & Ba — Adam: A Method for Stochastic Optimization (2015)](https://arxiv.org/abs/1412.6980)
- [Loshchilov & Hutter — Decoupled Weight Decay Regularization (2019)](https://arxiv.org/abs/1711.05101)
- [Smith & Topin — Super-Convergence (2019)](https://arxiv.org/abs/1708.07120)
- [Ruder — An overview of gradient descent optimization algorithms (2016)](https://arxiv.org/abs/1609.04747)
