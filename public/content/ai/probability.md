# Probability for Machine Learning

## Introduction

Probability theory is essential for machine learning. ML models are fundamentally about making predictions under uncertainty — whether it's predicting house prices (regression uncertainty), classifying images (prediction confidence), or learning from noisy data. This lesson covers the probability foundations every ML practitioner needs.

## Basic Probability

### Definitions
- **Experiment:** A process with uncertain outcome
- **Sample space $\Omega$:** The set of all possible outcomes
- **Event $A$:** A subset of outcomes
- **Probability $P(A)$:** A number between 0 and 1 quantifying likelihood

### Axioms (Kolmogorov)
1. $P(A) \geq 0$ for all events $A$
2. $P(\Omega) = 1$ (something always happens)
3. If $A \cap B = \emptyset$, then $P(A \cup B) = P(A) + P(B)$

### Conditional Probability
The probability of $A$ given that $B$ occurred:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Interpretation: $P(\text{model predicts correctly} | \text{actual class})$ — the core of classification metrics.

### Independence
Events $A$ and $B$ are independent iff:

$$P(A \cap B) = P(A) P(B)$$

Equivalently: $P(A|B) = P(A)$

## Bayes' Theorem

The most important formula in probabilistic ML:

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

In ML notation, using $H$ for hypothesis and $D$ for data:

$$P(H|D) = \frac{P(D|H) P(H)}{P(D)}$$

| Term | ML Meaning |
|------|------------|
| $P(H)$ — Prior | Our belief before seeing data |
| $P(D|H)$ — Likelihood | Probability of data given hypothesis |
| $P(H|D)$ — Posterior | Updated belief after seeing data |
| $P(D)$ — Evidence | Normalizing constant |

Bayesian inference updates priors to posteriors — this is the foundation of Bayesian machine learning.

## Random Variables

A random variable $X$ maps outcomes to real numbers.

### Discrete RVs
Takes countable values (0, 1, 2, ... or any distinct set).

**Probability Mass Function (PMF):** $P(X = x)$

### Continuous RVs
Takes uncountable values (any real number in a range).

**Probability Density Function (PDF):** $p(x)$ where $P(a \leq X \leq b) = \int_a^b p(x)dx$

### Expectation
The weighted average of a random variable:

| Type | Formula |
|------|---------|
| Discrete | $E[X] = \sum_x x \cdot P(X=x)$ |
| Continuous | $E[X] = \int_{-\infty}^{\infty} x \cdot p(x)dx$ |

Key properties: $E[aX + b] = aE[X] + b$

### Variance
Measures spread around the mean:

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

Standard deviation: $\sigma = \sqrt{\text{Var}(X)}$

## Common Distributions

### Bernoulli Distribution
For binary outcomes (0 or 1):
$$P(X=1) = p, \quad P(X=0) = 1-p$$
Mean: $p$, Variance: $p(1-p)$

Used for: single coin flip, binary classification labels.

### Gaussian (Normal) Distribution
The most important distribution in ML:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Key facts:
- ~68% of data within 1 std dev of mean
- ~95% within 2 std devs
- ~99.7% within 3 std devs

The Central Limit Theorem: sums of many independent random variables tend toward Gaussian — this is why Gaussian assumptions are so common.

### Categorical Distribution
Generalization of Bernoulli to $K$ categories:
$$P(X=k) = p_k, \quad \sum_{k=1}^K p_k = 1$$

Used for: multi-class classification, softmax outputs.

## Distributions in ML

| ML Context | Relevant Distributions |
|-----------|----------------------|
| Binary classification | Bernoulli (labels), Binomial (count of positives) |
| Multi-class classification | Categorical, Multinomial |
| Regression (continuous) | Gaussian |
| Counts/rates | Poisson |
| Probabilities | Beta, Dirichlet |

## Practice Problems

1. **If $P(A) = 0.3$, $P(B) = 0.4$, and $P(A \cap B) = 0.1$, find $P(A|B)$.**
   <details><summary>Answer</summary>$P(A|B) = P(A \cap B) / P(B) = 0.1 / 0.4 = 0.25$</details>

2. **A medical test for a disease has 99% sensitivity (P(test+|disease+) and 95% specificity (P(test-|disease-)). If 1% of the population has the disease, what is P(disease+|test+)?**
   <details><summary>Answer</summary>Bayes: $P(D+|T+) = \frac{P(T+|D+)P(D+)}{P(T+)} = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} \approx 0.167$ (only ~17%! This is the base rate fallacy.)</details>

3. **A Gaussian has $\mu = 100$ and $\sigma = 15$. What range contains roughly 95% of values?**
   <details><summary>Answer</summary>95% within $2\sigma$ of mean: $[100 - 2(15), 100 + 2(15)] = [70, 130]$</details>

## Key Takeaways

- Conditional probability $P(A|B)$ is the backbone of classification, confusion matrices, and metrics
- Bayes' theorem: $P(H|D) = P(D|H)P(H)/P(D)$ — prior + data = posterior
- Expectation $E[X]$ is the long-run average; variance $\text{Var}(X)$ measures spread
- Gaussian distribution is ubiquitous due to the Central Limit Theorem
- Many ML algorithms have probabilistic interpretations (Naive Bayes, logistic regression, etc.)
- Always check base rates — a highly accurate test can still have low positive predictive value when the condition is rare
