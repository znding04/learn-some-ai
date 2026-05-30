---
title: "Dependency Parsing and Semantic Role Labeling"
level: beginner
topic: ai-for-linguistics
order: 4
---

# Dependency Parsing and Semantic Role Labeling

## Overview

While constituency parsing organizes sentences into nested phrase structures, **dependency parsing** takes a different approach: it represents a sentence as a directed graph where every word (except the root) depends on exactly one other word — its **head**. The head is the syntactic governor, and the dependent is the modifier or argument.

The sentence *The cat sat on the mat* has this dependency structure:

```
      sat
    ┌──┼──────┐
    │  │      │
  The cat   on
           ┌──┼──┐
           │  │  the mat
           │  │
          the mat
```

*sat* is the root; *cat* is its subject (nsubj); *on* is the prepositional complement of *sat*; *mat* is the object of the preposition *on*.

Dependency parsing has several advantages over constituency parsing for NLP applications: it handles languages with flexible word order (Czech, Turkish), produces flat structures that map well to semantic representations, and the labeled arcs directly encode grammatical relations that are useful for downstream tasks.

---

## Grammatical Relations and Universal Dependencies

The **Universal Dependencies (UD)** project (Nivre et al., 2016+) defines a cross-linguistically consistent set of grammatical relations. UD v3 labels include:

| Label | Description | Example |
|-------|-------------|---------|
| nsubj | Nominal subject | *She* laughed |
| obj | Direct object | He ate *apples* |
| iobj | Indirect object | She gave *me* a book |
| nmod | Nominal modifier | the man *with a hat* |
| amod | Adjectival modifier | *big* dog |
| det | Determiner | *the* cat |
| case | Case marking | in *Warsaw* |
| root | Root of sentence | (virtual root) |

UD trees are **projective**: if there is an arc from head $h$ to dependent $d$, all words between $h$ and $d$ in the surface order must also be dependents of some node on the path between them. This constraint corresponds to the parser never needing to "skip over" unrelated words.

---

## Transition-Based Parsing

Transition-based parsers process sentences left-to-right, maintaining a stack of partially processed words and a buffer of remaining input words. At each step, they apply one of a small set of actions.

A standard arc-standard transition system for projective parsing:

- **Shift**: Move the first buffer word onto the stack
- **Left-Arc**: Create an arc from the top stack item ($s_1$) to the new stack top ($s_2$), and remove $s_2$. Used for dependencies where the new word is the head.
- **Right-Arc**: Create an arc from $s_2$ to $s_1$, and remove $s_1$.

```python
from collections import deque

class ArcStandardParser:
    def __init__(self, sentence):
        # sentence: list of (word, pos) tuples
        self.sentence = sentence
        self.stack = [(0, 'ROOT')]  # (index, form)
        self.buffer = deque(range(1, len(sentence) + 1))  # word indices
        self.arcs = []  # (head_idx, dep_idx, label)

    def step(self, action):
        """Apply a transition action"""
        if action == 'shift':
            word_idx = self.buffer.popleft()
            self.stack.append((word_idx, self.sentence[word_idx - 1][0]))
        elif action == 'left_arc':
            dep = self.stack.pop()
            head = self.stack[-1]
            self.arcs.append((head[0], dep[0], 'dep'))
        elif action == 'right_arc':
            head = self.stack.pop()
            dep = self.stack[-1]
            self.arcs.append((head[0], dep[0], 'dep'))

    def is_final(self):
        return len(self.buffer) == 0 and len(self.stack) == 1

# Neural transition-based parser (simplified)
class NeuralTransitionParser(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_relations=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.action_classifier = nn.Linear(hidden_dim * 3, 3)  # shift, la, ra
        self.rel_classifier     = nn.Linear(hidden_dim * 3, num_relations)

    def encode(self, word_ids, arcs=None):
        """BiLSTM encoder with arc representations"""
        embeds = self.embed(word_ids)
        hiddens, _ = self.rnn(embeds)
        return hiddens

    def predict_action(self, stack_repr, buffer_repr, context_repr):
        """Score transition actions given current parser state"""
        features = torch.cat([stack_repr[-1], buffer_repr[0], context_repr.mean(0)])
        return self.action_classifier(features)
```

---

## Graph-Based Parsing

Graph-based parsers treat the entire parse as a structure prediction problem. For a sentence of length $n$, the parser outputs a directed tree (or graph) over $n+1$ nodes (including the root at position 0).

**Eisner algorithm**: An $O(n^3)$ dynamic programming algorithm for projective dependency parsing that respects arc-factored scores — the score of a tree is the sum of individual arc scores. It uses a clever triangulation of the span to ensure projectivity.

The neural extension replaces arc scores with learned neural network outputs:

$$s(h, d) = \mathbf{w}^T \cdot \ MLP([\mathbf{h}_h; \mathbf{h}_d; \mathbf{h}_h \circ \mathbf{h}_d])$$

where $\mathbf{h}_i$ is the LSTM representation of word $i$, and $\circ$ denotes elementwise product.

---

## Semantic Role Labeling (SRL)

Once we have syntactic structure, we can ask: who did what to whom, when, where, and how? **Semantic Role Labeling (SRL)** identifies **predicate-argument structures** — which verb (predicate) takes which arguments, and what semantic role each argument plays.

The dominant framework uses PropBank (Palmer et al., 2005), which defines frame-specific roles:

- **Arg0**: Proto-agent (typically the actor) — *She* ate
- **Arg1**: Proto-patient (typically the undergoer) — She ate *an apple*
- **Arg2**: Benefactive, instrument, attribute — She ate *with a fork*
- **ArgM-TMP**: Temporal modifier — She ate *at noon*
- **ArgM-LOC**: Location — She ate *in Paris*

```python
# Simplified SRL as BIO tagging over syntactic constituents
class SemanticRoleLabeler(nn.Module):
    """
    Given a sentence with identified predicates and syntactic parse,
    predict semantic roles for each constituent.
    """
    def __init__(self, hidden_dim, num_roles):
        super().__init__()
        self.role_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # constituent + predicate + context
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles)
        )

    def get_roles(self, constituent_repr, predicate_repr, sent_repr):
        """Classify role for one constituent given a predicate"""
        combined = torch.cat([
            constituent_repr,
            predicate_repr,
            constituent_repr * predicate_repr,
            sent_repr.mean(0)
        ], dim=-1)
        return self.role_classifier(combined)

# BIO tagging for span-level SRL
BIO_tags = {
    'O': 0,       # Outside
    'B-ARG0': 1, 'I-ARG0': 2,
    'B-ARG1': 3, 'I-ARG1': 4,
    'B-ARG2': 5, 'I-ARG2': 6,
    'B-ARGM': 7, 'I-ARGM': 8,
}
# A sentence "She ate an apple with a fork" predicate="ate"
# BIO sequence: B-ARG0 O B-ARG1 B-ARGM O
```

---

## Key Concepts

- **Dependency parsing**: Every word has exactly one head (except the root), producing a directed tree
- **Universal Dependencies (UD)**: A cross-linguistic framework for annotating grammatical relations
- **Projectivity**: The constraint that arcs don't cross; valid for UD-style treebanks
- **Transition-based parsing**: Left-to-right stack-based parsing with Shift/Left-Arc/Right-Arc actions
- **Graph-based parsing**: Global structure prediction using arc-factored or fully factorized scores
- **Semantic Role Labeling (SRL)**: Identifying predicate-argument structure using PropBank-style roles
- **Eisner algorithm**: $O(n^3)$ dynamic programming for projective dependency parsing

## Exercises

1. **Dependency analysis**: Draw the dependency parse for the sentence *The quick brown fox jumps over the lazy dog.* Label each arc with its UD relation.
2. **Transition sequence**: For the sentence *The cat slept*, show the full transition sequence using arc-standard transitions that produces nsubj(cat, The) and root(slept, cat).
3. **SRL frame**: For the verb *gave* in *John gave Mary a book yesterday*, identify Arg0, Arg1, Arg2, and ARGM-TMP.

## Further Reading

- Nivre, J. et al. (2016). "Universal Dependencies v1." *LREC*.
- Dozat, T. & Manning, C.D. (2017). "Deep Biaffine Attention for Neural Dependency Parsing." *ICLR*.
- He, L. et al. (2017). "Syntax-aware Neural Semantic Role Labeling." *ACL 2017*.
- Palmer, M. et al. (2005). "The Proposition Bank: An Annotated Corpus of Semantic Roles." *Computational Linguistics* 31(1).
