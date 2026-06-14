---
title: "Language Modeling as Linguistic Knowledge Acquisition"
difficulty: intermediate
topic: ai-for-linguistics
order: 6
estimatedTime: "30 minutes"
summary: "Examines how language models trained on next-word prediction implicitly acquire linguistic knowledge, from n-gram models to neural LMs, and what probing studies reveal about their syntactic and semantic competence."
---

# Language Modeling as Linguistic Knowledge Acquisition

## Overview

A language model (LM) assigns a probability distribution over sequences of words. Given a prefix $w_1, w_2, \ldots, w_{t-1}$, it predicts the probability of each possible next word $w_t$:

$$P(w_t | w_1, \ldots, w_{t-1})$$

At first glance, this seems like a simple statistical task. But consider: to accurately predict the next word, a language model must implicitly represent:

- **Syntactic constraints**: *The cat sat on **the*** makes grammatical sense; *The cat sat on **and*** does not.
- **Semantic coherence**: *She drank a cup of **coffee*** is more likely than *She drank a cup of **telephone***.
- **World knowledge**: To predict *He turned the key and the car **started*** requires knowing about how cars work.
- **Coreference and discourse**: *John walked into the room. **He** sat down.* — knowing *He* refers to *John*.

This raises a profound question: **does training on next-word prediction teach language models something like linguistic competence?** Or are they just sophisticated pattern matchers? This question sits at the intersection of AI and linguistic theory.

---

## N-gram Language Models

The simplest LM is an **n-gram model**: it conditions on the previous $n-1$ words using the chain rule:

$$P(w_1, \ldots, w_T) = \prod_{t=1}^T P(w_t | w_{t-n+1}, \ldots, w_{t-1})$$

Counts are collected from a corpus, and probabilities are estimated via maximum likelihood estimation (MLE):

$$P_{\text{MLE}}(w_t | w_{t-n+1}^{t-1}) = \frac{\text{count}(w_{t-n+1}^{t-1}, w_t)}{\text{count}(w_{t-n+1}^{t-1})}$$

**Backoff** and **smoothing** (Kneser-Ney, Laplace) handle unseen n-grams.

```python
class NGramLM:
    """Simple n-gram language model with Laplace smoothing"""
    def __init__(self, n):
        self.n = n
        self.counts = {}
        self.context_counts = {}

    def train(self, sentences):
        """Build n-gram counts from a list of sentences"""
        for sent in sentences:
            tokens = ['<s>'] * (self.n - 1) + sent + ['</s>']
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                word    = tokens[i]
                self.counts[(context, word)] = self.counts.get((context, word), 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1

    def prob(self, word, context):
        """P(word | context) with Laplace smoothing"""
        ctx = tuple(context[-(self.n - 1):])
        V   = len(set(w for (_, w) in self.counts.keys()))  # vocab size
        alpha = 1.0  # Laplace smoothing
        c = self.counts.get((ctx, word), 0)
        denom = self.context_counts.get(ctx, 0) + alpha * V
        return (c + alpha) / denom

    def score(self, sentence):
        """Log probability of a sentence"""
        tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        logp = 0.0
        for i in range(self.n - 1, len(tokens)):
            ctx  = tokens[i - self.n + 1:i]
            word = tokens[i]
            logp += np.log(self.prob(word, ctx))
        return logp

# Perplexity: exp(-1/T * sum log P(w_t | w_{<t}))
def perplexity(lm, test_sentences):
    total_logp = 0
    total_words = 0
    for sent in test_sentences:
        total_logp  += lm.score(sent)
        total_words += len(sent)
    return np.exp(-total_logp / total_words)
```

**Perplexity** is the standard evaluation metric — lower is better. A model with perplexity $PP = 20$ is as uncertain as if it were choosing uniformly among 20 words at each step.

---

## Neural Language Models

**Feedforward neural LM** (Bengio et al., 2003): The first neural LM, predicting the next word from the previous $n-1$ words via a learned embedding layer and a softmax output:

$$P(w_t | w_{t-n+1}^{t-1}) = \text{softmax}(\mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{e}_{t-n+1}^{t-1} + \mathbf{b}_1) + \mathbf{b}_2)$$

**RNN/LSTM language models**: By using recurrent hidden states, RNN LMs condition on the entire history without the fixed context window limitation:

$$\mathbf{h}_t = \sigma(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)$$
$$P(w_t | w_{<t}) = \text{softmax}(\mathbf{W}_{ho} \mathbf{h}_t + \mathbf{b}_o)$$

LSTMs and GRUs add gating mechanisms to mitigate vanishing gradients, enabling learning of long-range dependencies.

---

## What Do Language Models Learn About Grammar?

### Syntactic Knowledge

GPT-2 can correctly complete number agreement constructions:
- Input: *"The keys to the cabinet are on the **table**"* ✓
- Input: *"The key to the cabinet is on the **table**"* ✓

This requires the model to track subject-verb agreement over an intervening PP modifier (*to the cabinet*). But does the model "know" the rule, or has it memorized similar examples?

**Probing studies** (Liu et al., 2019; Hewitt & Liang, 2019) train classifiers on the model's internal representations to extract syntactic structure. If a linear classifier can predict subject-verb number from the hidden state at the verb position with high accuracy, the model encodes this information. Such studies show that transformers encode substantial syntactic information, particularly in their attention heads.

### The Surprising Capabilities of Large LMs

 GPT-3 (Brown et al., 2020) demonstrated that scaling a decoder-only LM to 175 billion parameters yields emergent few-shot capabilities — the ability to perform new tasks from just a few examples in the prompt, without gradient updates.

| Capability | Example |
|-----------|---------|
| Grammar correction | *"I goes to the store."* → *"I go to the store."* |
| Word in context (WiC) | *"The bank of the river..."* vs. *"The bank promised..."* |
| Coreference resolution | *John gave Mary a book. **She** thanked **him**.* |
| Translation (limited) | *"The cat sat on the mat"* → French *"Le chat s'est assis..."* |

But these capabilities have limits. Systematic generalization — the ability to apply a rule to novel combinations — remains challenging even for large models.

---

## The Linguistic Collapse Phenomenon

Recent research (arXiv:2605.28826, ACL 2026) has identified a troubling phenomenon: **training objectives can cause linguistic collapse**. When language models are trained primarily on next-token prediction at massive scale, certain linguistic distinctions erode. For example:

- **Stylistic variation** collapses: formal vs. informal registers become indistinguishable
- **Dialect variation** flattens: regional and socioeconomic accents are normalized away
- **Pragmatic distinctions** weaken: implicature, irony, and register become harder to detect

This suggests that the training objective (next-token prediction of web text) may not preserve all dimensions of linguistic knowledge equally. Different objectives — such as contrastive learning, interleaved with denoising — may be needed to maintain full linguistic richness.

---

## Key Concepts

- **Language model**: A probability distribution $P(w_t | w_1, \ldots, w_{t-1})$ over next tokens
- **Perplexity**: $\exp(-\frac{1}{T}\sum_t \log P(w_t | w_{<t}))$ — measures LM uncertainty
- **N-gram model**: Markov approximation conditioning on $n-1$ previous words
- **Neural LM**: Feedforward, RNN/LSTM, or Transformer-based next-token prediction
- **Probing**: Training a classifier on frozen LM representations to test what knowledge is encoded
- **Linguistic collapse**: Degradation of fine-grained linguistic distinctions (register, dialect, pragmatics) under certain training regimes
- **Systematic generalization**: The ability to apply a learned rule to novel combinations

## Exercises

1. **Perplexity comparison**: Train an n-gram LM and an LSTM LM on the same corpus. Compute perplexity on a held-out test set. Which performs better and why?
2. **Agreement probing**: Design a probing experiment to test whether a transformer LM encodes subject-verb agreement. What layer do you expect to be most informative, and why?
3. **Linguistic collapse**: Read arXiv:2605.28826 (abstract). How does the paper's finding about training objectives affect our understanding of what LLMs learn about language?

## Further Reading

- Bengio, Y. et al. (2003). "A Neural Probabilistic Language Model." *JMLR* 3, 1137–1155.
- Brown, T.B. et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
- Liu, N.F. et al. (2019). "Linguistic Knowledge and Transferability of Contextual Representations." *NAACL*.
- Hewitt, J. & Liang, P. (2019). "Designing and Interpreting Probes with Control Tasks." *EMNLP*.
- Gulordava, K. et al. (2018). "Colorless Green Recurrent Networks Dream Hierarchically." *NAACL*.
- "From Context Shift to Stylistic Collapse" (arXiv:2605.28826, ACL 2026).
