# What is Machine Learning?

## Definition

**Machine learning (ML)** is a subset of artificial intelligence where computers learn patterns from data — rather than being explicitly programmed with rules.

Traditional programming:
$$\text{data} + \text{program} \rightarrow \text{output}$$

Machine learning:
$$\text{data} + \text{output} \rightarrow \text{program}$$

## Three Types of ML

### 1. Supervised Learning

Learn from labeled examples (input → correct output).

- **Classification:** Predict a category (spam/not spam)
- **Regression:** Predict a continuous number (house price)

### 2. Unsupervised Learning

Find patterns in unlabeled data.

- **Clustering:** Group similar items (customer segmentation)
- **Dimensionality Reduction:** Compress features (PCA)

### 3. Reinforcement Learning

Learn by trial and error, receiving rewards or penalties.

- Used in robotics, game AI, recommendation systems.

## Key Concepts

- **Model:** The mathematical representation being learned
- **Training:** The process of adjusting model parameters to minimize error
- **Loss function:** Measures how wrong predictions are
- **Overfitting:** Model memorizes training data, fails on new data
- **Generalization:** Ability to perform well on unseen data

## Practice Reflection

1. **Would you use supervised or unsupervised learning to detect anomalous network traffic?**
   <details><summary>Answer</summary>Supervised (labeled normal/anomaly) if labels exist; unsupervised if no labels (anomaly detection / clustering approach).</details>
