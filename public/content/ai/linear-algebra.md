# Linear Algebra for ML

## Introduction

Linear algebra is the mathematical backbone of machine learning. Neural networks, principal component analysis, regression, and virtually every ML algorithm ultimately reduce to linear algebra operations on vectors and matrices. This lesson covers the essential concepts.

## Vectors

A vector is an ordered array of numbers — representing both a point in space and a direction.

### Representation
$$\vec{a} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix}$$

In ML, vectors often represent:
- A data point with $n$ features
- Model weights
- activations

### Vector Operations

| Operation | Formula | Result |
|-----------|---------|--------|
| Addition | $\vec{a} + \vec{b}$ | Component-wise sum |
| Scalar multiplication | $c\vec{a}$ | Scale each component |
| Dot product | $\vec{a} \cdot \vec{b} = \sum a_i b_i$ | Scalar |
| Magnitude | $\|\vec{a}\| = \sqrt{\sum a_i^2}$ | Length |

### Dot Product
The dot product measures similarity between vectors:

$$\vec{a} \cdot \vec{b} = \|\vec{a}\|\ \|\vec{b}\|\ \cos\theta$$

Key property: $\vec{a} \cdot \vec{b} = 0$ means orthogonal (perpendicular).

In ML, the dot product weights inputs: $z = \vec{w} \cdot \vec{x} + b$

## Matrices

A matrix is a 2D array of numbers with dimensions $m \times n$ (rows × columns).

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

### Matrix Multiplication
For $C = AB$ where $A$ is $m \times k$ and $B$ is $k \times n$:

$$C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}$$

Each element $C_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of $B$.

### Key Properties
- **Not commutative:** $AB \neq BA$ in general
- **Associative:** $(AB)C = A(BC)$
- **Identity matrix:** $AI = IA = A$
- **Transpose:** $(AB)^T = B^T A^T$

## Eigenvalues and Eigenvectors

An eigenvector of matrix $A$ is a non-zero vector $\vec{v}$ that only gets scaled (not rotated) when multiplied by $A$:

$$A\vec{v} = \lambda\vec{v}$$

where $\lambda$ is the corresponding **eigenvalue**.

### Why Eigenvalues Matter in ML

| Application | Role of Eigenvalues/vectors |
|-------------|----------------------------|
| PCA | eigenvectors of covariance matrix capture principal components |
| Spectral clustering | eigenvalues determine cluster assignments |
| PageRank | dominant eigenvector gives page rankings |
| Markov chains | stationary distributions are eigenvectors |

### Characteristic Equation
Find eigenvalues by solving:
$$\det(A - \lambda I) = 0$$

## Matrix in ML Context

### Linear Transformation
$y = W\vec{x} + \vec{b}$ where:
- $W$ is the weight matrix ($n_{\text{out}} \times n_{\text{in}}$)
- $\vec{x}$ is the input vector
- $\vec{b}$ is the bias vector

This is the core operation in a neural network layer.

### Data as Matrices
- Rows = samples, Columns = features
- $X \in \mathbb{R}^{n_{\text{samples}} \times n_{\text{features}}$

## Practice Problems

1. **Compute the dot product of $\vec{a} = [1, 3, -2]$ and $\vec{b} = [4, -1, 2]$**
   <details><summary>Answer</summary>$\vec{a} \cdot \vec{b} = 1(4) + 3(-1) + (-2)(2) = 4 - 3 - 4 = -3$</details>

2. **If $W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $\vec{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$, compute $W\vec{x}$**
   <details><summary>Answer</summary>$W\vec{x} = \begin{bmatrix} 1(5)+2(6) \\ 3(5)+4(6) \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$</details>

3. **Find eigenvalues of $A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$**
   <details><summary>Answer</summary>$\det(A - \lambda I) = (2-\lambda)(3-\lambda) - 0 = (2-\lambda)(3-\lambda) = 0$. So $\lambda_1 = 2, \lambda_2 = 3$.</details>

## Key Takeaways

- Vectors: points/directions in space; dot product measures similarity
- Matrices: represent linear transformations; matrix multiplication is row × column
- Eigenvectors $\vec{v}$ are scaled (not rotated) by $A$: $A\vec{v} = \lambda\vec{v}$
- Neural networks: $y = W\vec{x} + \vec{b}$ is pure linear algebra
- PCA, embeddings, and transformations all rely on eigenvectors of matrices
