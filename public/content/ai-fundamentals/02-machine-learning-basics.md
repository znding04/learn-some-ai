---
title: "Machine Learning Basics"
level: beginner
topic: ai-fundamentals
order: 2
---

# Machine Learning Basics

## Overview

Machine learning is the engine behind modern AI. Instead of writing explicit rules, we let algorithms discover patterns in data. This lesson covers the fundamental paradigms, core workflow, and key pitfalls every ML practitioner must understand.

### The Three Paradigms

**Supervised Learning** uses labeled data — each input comes with the correct answer. The model learns a function $f: X \rightarrow Y$ that maps inputs to outputs.

- **Classification**: Predict a discrete label. Is this email spam or not? $y \in \{0, 1\}$
- **Regression**: Predict a continuous value. What will this house sell for? $y \in \mathbb{R}$

**Unsupervised Learning** works with unlabeled data. The model finds hidden structure on its own.

- **Clustering**: Group similar data points (K-means, DBSCAN)
- **Dimensionality Reduction**: Compress high-dimensional data while preserving structure (PCA, t-SNE)
- **Density Estimation**: Learn the underlying probability distribution

**Reinforcement Learning (RL)** is about learning through interaction. An agent takes actions in an environment, receives rewards or penalties, and learns a policy $\pi(s) \rightarrow a$ that maximizes cumulative reward.

### The ML Workflow

Every ML project follows roughly this pipeline:

1. **Collect data** — Gather relevant, representative data
2. **Preprocess** — Clean, normalize, handle missing values
3. **Split data** — Divide into training set, validation set, and test set
4. **Train model** — Fit the model to training data
5. **Evaluate** — Measure performance on held-out test data
6. **Iterate** — Tune hyperparameters, try different models

### Train/Test Split

Never evaluate a model on the data it was trained on. The standard approach:

$$\text{Dataset} = \underbrace{\text{Train (70-80\%)}}_{\text{learn from}} + \underbrace{\text{Validation (10-15\%)}}_{\text{tune on}} + \underbrace{\text{Test (10-15\%)}}_{\text{final eval}}$$

The test set must remain untouched until the very end — it's your unbiased estimate of real-world performance.

### Overfitting vs Generalization

The central tension in ML:

- **Overfitting**: The model memorizes the training data, including noise. It performs perfectly on training data but poorly on new data.
- **Underfitting**: The model is too simple to capture the underlying pattern.
- **Generalization**: The sweet spot — the model learns the true pattern and performs well on unseen data.

Overfitting is the more common danger. Signs include a large gap between training accuracy (high) and test accuracy (low).

### Loss Functions

A loss function $L(y, \hat{y})$ measures how wrong predictions $\hat{y}$ are compared to true labels $y$. Common choices:

- **Mean Squared Error** (regression): $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **Cross-Entropy** (classification): $L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$

The goal of training is to minimize this loss.

### Gradient Descent

Gradient descent is the optimization algorithm that powers most ML. The idea: compute the gradient (direction of steepest increase) of the loss function, then take a step in the opposite direction.

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where:
- $\theta$ = model parameters (weights)
- $\eta$ = learning rate (step size)
- $\nabla L$ = gradient of the loss function

If the learning rate is too large, you overshoot. Too small, and training takes forever.

## Key Concepts

- **Supervised Learning**: Learning from labeled input-output pairs
- **Unsupervised Learning**: Finding hidden patterns in unlabeled data
- **Reinforcement Learning**: Learning through trial, error, and rewards
- **Overfitting**: Model memorizes training data, fails to generalize
- **Loss Function**: Quantifies prediction error; training minimizes this
- **Gradient Descent**: Iterative optimization by following the negative gradient

## Code Examples

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + np.random.randn(100) * 2

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"Learned slope: {model.coef_[0]:.2f}, intercept: {model.intercept_:.2f}")
```

## Diagrams

```
Overfitting vs Underfitting:

  Error
    │
    │ ╲  Test Error
    │  ╲___________╱
    │        ╲    ╱
    │         ╲╱   ← Sweet spot
    │   ___________
    │  ╱  Training Error
    │╱
    └──────────────────── Model Complexity
   Underfit           Overfit
```

## Exercises

1. **Classify the task**: Is predicting tomorrow's temperature supervised or unsupervised? Classification or regression?
2. **Code challenge**: Modify the code above to use a polynomial regression (degree 10). What happens to the train vs test MSE? Why?
3. **Thought experiment**: You have 50 data points. How would you split them into train/validation/test? Why might cross-validation be better here?

## Further Reading

- Andrew Ng's Machine Learning course (Stanford CS229)
- Hastie, T., Tibshirani, R., & Friedman, J. *The Elements of Statistical Learning*
- scikit-learn documentation: https://scikit-learn.org/stable/
