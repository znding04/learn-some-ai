---
title: "Word Embeddings and Beyond: Representing Linguistic Knowledge"
difficulty: intermediate
topic: ai-for-linguistics
order: 5
estimatedTime: "30 minutes"
summary: "Traces the evolution of word representations from one-hot vectors through dense embeddings (word2vec, GloVe) to contextual representations (ELMo, BERT) that capture polysemy and context-dependence."
---

# Word Embeddings and Beyond: Representing Linguistic Knowledge

## Overview

One of the central challenges in NLP is representing meaning computationally. Words are discrete symbols — *cat* and *dog* are just strings — but their meanings are related in complex ways. *Cat* and *dog* are both animals, both pets, both nouns, and yet distinct. How do we represent these relationships mathematically?

The breakthrough came in 2013 with the **word2vec** paper (Mikolov et al.), which showed that the meanings of words could be captured as dense vectors in a continuous space — **embeddings** — where similar words cluster together and meaningful directions in the space encode semantic relationships.

This lesson traces the evolution of word representations: from one-hot vectors and co-occurrence matrices, through dense embeddings (word2vec, GloVe, fastText), to contextual representations (ELMo, BERT) that capture polysemy and context-dependence.

---

## From One-Hot to Dense Embeddings

### One-Hot Encoding

In a one-hot representation, each word in a vocabulary $V$ is represented as a $|V|$-dimensional vector with a single 1 and zeros everywhere else:

$$\mathbf{v}_{\text{cat}} = [0, 0, 1, 0, \ldots, 0, 0]$$
$$\mathbf{v}_{\text{dog}} = [0, 0, 0, 1, \ldots, 0, 0]$$

These vectors are orthogonal — their dot product is always zero. This makes them mathematically clean but semantically meaningless: *cat* and *dog* are no more similar than *cat* and *refrigerator*.

### Word-Document Matrices and Co-occurrence Matrices

A more informative representation exploits co-occurrence: words that appear in similar contexts tend to have similar meanings (the **distributional hypothesis**). A word-document matrix $M$ has rows for words and columns for documents; $M_{ij}$ is the count of word $i$ in document $j$. A word-context matrix can use sliding windows, dependency relations, or other contexts.

These sparse, high-dimensional matrices can be reduced via **Singular Value Decomposition (SVD)**:

$$M = U \Sigma V^T$$

Taking the top $k$ singular values gives a $k$-dimensional dense approximation: $\hat{M} = U_k \Sigma_k V_k^T$.

---

## word2vec: Dense Embeddings from Local Context

The word2vec insight was to train a shallow neural network (two layers, no activation on output) to predict context words given a target word (Skip-gram) or vice versa (CBOW). The learned hidden-layer weights become the word embeddings.

**Skip-gram** maximizes:
$$\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$$

where $c$ is the context window radius. With softmax:

$$p(w_j | w_i) = \frac{\exp(\mathbf{u}_j^T \mathbf{v}_i)}{\sum_{k=1}^{|V|} \exp(\mathbf{u}_k^T \mathbf{v}_i)}$$

This is prohibitively expensive for large vocabularies, so **negative sampling** is used instead: for each true (target, context) pair, sample $k$ negative words from a noise distribution and maximize:

$$\log \sigma(\mathbf{u}_j^T \mathbf{v}_i) + \sum_{n=1}^{k} \mathbb{E}_{w_n \sim P_n(w)} \left[ \log \sigma(-\mathbf{u}_n^T \mathbf{v}_i) \right]$$

where $\sigma(x) = 1/(1 + e^{-x})$.

```python
# Minimal word2vec skip-gram implementation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_emb = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, context, neg_samples):
        """
        target: (batch,) — indices of target words
        context: (batch,) — indices of positive context words
        neg_samples: (batch, k) — indices of negative samples
        """
        target_emb  = self.target_emb(target)          # (B, D)
        pos_scores  = torch.sum(target_emb * self.context_emb(context), dim=-1)
        pos_loss    = -torch.nn.functional.logsigmoid(pos_scores)

        neg_emb     = self.context_emb(neg_samples)    # (B, K, D)
        neg_scores  = torch.bmm(neg_emb, target_emb.unsqueeze(-1)).squeeze(-1)  # (B, K)
        neg_loss    = -torch.nn.functional.logsigmoid(-neg_scores).sum(dim=-1)

        return (pos_loss + neg_loss).mean()

# After training: self.target_emb.weight gives the learned word vectors
```

---

## GloVe: Global Co-occurrence Statistics

GloVe (Pennington et al., 2014) combines the intuition of co-occurrence matrices with the efficiency of word2vec. It trains on global word-word co-occurrence counts and learns embeddings such that:

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j = \log X_{ij}$$

where $X_{ij}$ is the co-occurrence count of words $i$ and $j$, and $b_i, \tilde{b}_j$ are bias terms. The objective is a weighted least squares:

$$\mathcal{L} = \sum_{i,j} f(X_{ij}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

where $f$ is a clipping function that downweights rare and very frequent co-occurrences.

---

## Evaluating Embeddings

**Intrinsic evaluation** tests embeddings directly: word similarity (how close is *cat* to *dog*?), analogy completion (*king - man + woman = ?*), and categorization (do embeddings cluster by semantic category?).

**Extrinsic evaluation** tests embeddings in downstream tasks: using them as features in a POS tagger, parser, or sentiment classifier and measuring task performance.

The famous word2vec analogy results:
$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$
$$\mathbf{v}_{\text{Athens}} - \mathbf{v}_{\text{Greece}} + \mathbf{v}_{\text{Italy}} \approx \mathbf{v}_{\text{Rome}}$$

---

## Contextual Embeddings: ELMo and BERT

Static word embeddings assign the same vector to a word regardless of context — but words are polysemous. *Bank* means different things in *river bank* and *financial bank*. **Contextual embeddings** compute a representation that depends on the surrounding words.

**ELMo** (Peters et al., 2018) uses a bidirectional LSTM (biLM) trained on a language modeling objective. Each word's representation is a weighted combination of the forward and backward LSTM states at that position. The learned weights indicate which layer best captures different phenomena (lower layers capture syntax, higher layers capture semantics).

**BERT** (Devlin et al., 2019) uses a Transformer encoder and introduces the **masked language modeling (MLM)** objective: 15% of tokens are [MASK], and the model must predict them from context. This bidirectional context contrasts with GPT's unidirectional language modeling.

$$p(x_i | x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n) = \text{softmax}(\mathbf{W}_o \mathbf{h}_i)$$

```python
# Using pre-trained BERT for contextual embeddings
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = BertModel.from_pretrained('bert-base-uncased')

sentence = "The bank of the river was muddy."
tokens   = tokenizer.encode(sentence, return_tensors='pt')

with torch.no_grad():
    outputs = model(tokens)
    # last_hidden_state: (batch, seq_len, hidden_dim)
    # pooler_output: [CLS] token after linear + tanh
    contextual_embeds = outputs.last_hidden_state[0]  # first example

# Each word now has a unique embedding conditioned on its context
# 'bank' in 'financial bank' vs 'river bank' will have different vectors
```

---

## Key Concepts

- **Distributional hypothesis**: Words that occur in similar contexts have similar meanings
- **Embedding**: A dense real-valued vector representing a word or token
- **word2vec Skip-gram**: Predicting context words from target word; trained via negative sampling
- **GloVe**: Global co-occurrence statistics + factorization, combining count and prediction methods
- **ELMo**: Deep biLSTM biLM producing layered contextual representations
- **BERT**: Bidirectional Transformer encoder with masked language modeling
- **Polysemy**: The property of having multiple related meanings; addressed by contextual embeddings

## Exercises

1. **Analogy experiments**: Use pre-trained word2vec or GloVe embeddings. Compute the analogy *Paris - France + Italy*. What do you get? Try 3 other morphological or semantic analogies.
2. **Embedding evaluation**: Evaluate GloVe vs. BERT embeddings on a word-in-context task (WiC). What differences do you observe?
3. **Polysemy analysis**: Find 5 English words with distinct meanings in different contexts. For each, extract BERT contextual embeddings from two different sentences and compute their cosine similarity.

## Further Reading

- Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space." *ICLR Workshop*.
- Pennington, J. et al. (2014). "GloVe: Global Vectors for Word Representation." *EMNLP*.
- Peters, M. et al. (2018). "Deep Bidirectional Transformers for Language Understanding." *NAACL*.
- Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*.
- Ethayarajh, K. (2019). "Contextual Embeddings: When Are They Worth More Than Word Embeddings?" *NAACL*.
