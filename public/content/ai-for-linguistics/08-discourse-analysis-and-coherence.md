---
title: "Discourse Analysis and Coherence"
level: intermediate
topic: ai-for-linguistics
order: 8
---

# Discourse Analysis and Coherence

## Overview

Individual sentences don't exist in isolation — they form texts, conversations, and arguments. **Discourse analysis** studies how sentences connect to create larger units of meaning: how speakers introduce and track referents across sentences, how they signal logical relations between clauses, and how they structure arguments and narratives.

A text like *"John bought a new car. He drove it to work"* has a property that pure sentence-level NLP misses: the *pronoun* *he* refers back to *John*, and *it* refers to *the car*. Without tracking these **referential relations**, we cannot fully understand the text.

At a larger scale, coherence is what makes a text feel like a text rather than a random collection of sentences. A coherent paragraph on a single topic with logical transitions feels organized; a disjointed one jumps randomly between topics. Modeling this has been a central challenge for computational linguistics.

---

## Rhetorical Structure Theory (RST)

**Rhetorical Structure Theory** (Mann & Thompson, 1988) analyzes text into a hierarchy of discourse units connected by rhetorical relations. The two fundamental units are:

- **Nucleus**: The more essential unit — the main point
- **Satellite**: The supporting unit — elaboration, cause, condition, etc.

RST relations include:

| Relation | Nucleus | Satellite | Example |
|----------|---------|---------|---------|
| **Elaboration** | Main idea | Details | *"The cat sat. The cat sat on the mat."* |
| **Cause** | Effect | Reason | *"It rained. The ground was wet."* |
| **Contrast** | Item A | Item B | *"John likes coffee. Mary prefers tea."* |
| **Condition** | Main | Hypothetical | *"If it rains, we'll cancel."* |
| **Background** | Event | Context | *"The door opened. It had been unlocked."* |
| **Attribution** | Content | Source | *"He won. According to the newspaper..."* |

The discourse parse is a tree, similar to syntactic parse but spanning larger textual units.

```python
# Discourse relation classification
class RSTClassifier(nn.Module):
    """
    Classify discourse relations between two text spans.
    Input: vector representations of nucleus and satellite spans
    Output: discourse relation label
    """
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # [nuc; sat; nuc⊙sat; nuc-sat]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_relations)
        )

    def forward(self, nucleus_repr, satellite_repr):
        combined = torch.cat([
            nucleus_repr,
            satellite_repr,
            nucleus_repr * satellite_repr,      # Hadamard product
            nucleus_repr - satellite_repr        # Difference
        ], dim=-1)
        return self.classifier(combined)

# RST parsing: build discourse tree bottom-up using shift-reduce
class RSTParser:
    def __init__(self):
        self.stack = []   # EDUs (Elementary Discourse Units)
        self.buffer = []  # Unprocessed EDUs

    def shift(self):
        self.stack.append(self.buffer.pop(0))

    def reduce_nuclear(self, relation):
        satellite = self.stack.pop()
        nucleus    = self.stack.pop()
        self.stack.append(('NUC', nucleus, satellite, relation))

    def reduce_satellite(self, relation):
        satellite = self.stack.pop()
        nucleus   = self.stack[-1]  # peek
        # Replace nucleus with new node containing both
        self.stack[-1] = ('NUC', nucleus, satellite, relation)
```

---

## Entity Tracking and Coreference Resolution

**Coreference resolution** identifies when two expressions refer to the same entity. In:

*"Mary met Susan at the café. **She** ordered a latte. **She** paid and **they** left,"*

*She₁* and *She₂* both refer to *Mary* and *Susan* (who?), and *they* refers to both together.

### Mention Detection

First, identify all **mentions** — noun phrases that refer to discourse entities:
- Pronominal mentions: *he, she, it, they, this, that*
- Named entity mentions: *Mary, Susan, the café*
- Definite noun phrases: *the café, the latte*
- Bare plurals: *cats* (in some contexts)

### Coreference Systems

Modern coreference resolvers use span-based neural models (Lee et al., 2017, 2018):

1. **Mention proposals**: Score all spans $(i, j)$ as potential mentions using a feedforward network over contextual embeddings.
2. **Mention clustering**: For each pair of proposed mentions $(m_1, m_2)$, compute a coreference score based on their representations and antecedent features.
3. **Clustering**: Use a clustering algorithm (pruning + nearest-neighbor) to form the final coreference chains.

```python
class CoreferenceResolver(nn.Module):
    """
    Span-based coreference model (Lee et al., 2017)
    """
    def __init__(self, embed_dim, hidden_dim, num_genres):
        super().__init__()
        self.mention_proposer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.attention = nn.Linear(embed_dim, 1)

    def score_antecedent(self, span_repr, antecedent_repr, features):
        """Score how likely span is co-referential with antecedent"""
        pair_repr = torch.cat([
            span_repr * antecedent_repr,     # interaction
            torch.abs(span_repr - antecedent_repr),  # distance
            features
        ], dim=-1)
        return self.antecedent_scorer(pair_repr)

    def forward(self, sentence_embeds, genre_ids):
        # Step 1: propose mentions (all subspans above a score threshold)
        mentions = self.propose_mentions(sentence_embeds)

        # Step 2: for each mention, score possible antecedents
        antecedent_scores = {}
        for m in mentions:
            for a in mentions[:m]:
                antecedent_scores[(a, m)] = self.score_antecedent(
                    m.repr, a.repr, compute_features(m, a, genre_ids)
                )

        # Step 3: clustering via greedy antecedent selection
        clusters = self.cluster(mentions, antecedent_scores)
        return clusters
```

---

## Coherence Models

What makes a text feel coherent beyond local relations? At the discourse level, coherence involves:

1. **Topic continuity**: Successive sentences should share topics. Sudden topic shifts without signaling feel jarring.
2. **Argument structure**: In persuasive texts, claims should be supported by evidence; cause-effect relations should be clearly signaled.
3. **Information flow**: Given-new contract — sentences should introduce new information relative to what the reader already knows.

### Entity Grid Model

The **Entity Grid Model** (Barzilay & Lapata, 2008) represents a text as a matrix where rows are sentences, columns are entity mentions, and cell values indicate the grammatical role of each entity in each sentence (subject, object, other, absent):

```
         John  Mary  car
S1:      SUBJ  ---   OBJ    (John bought the car)
S2:      ---   SUBJ  SUBJ    (Mary drove it)
S3:      SUBJ  ---   ---     (John was happy)
```

Transitions between grammatical roles (subject→object, subject→subject) are scored, and the overall coherence of the text is computed as the sum of transition probabilities from a Markov chain model.

### Neural Coherence Models

More recent approaches use neural sequence models to score coherence:

$$s(\text{text}) = \prod_{t=1}^T P(\text{sent}_t | \text{sent}_{<t}; \theta)$$

A hierarchical RNN encodes each sentence, then a second RNN models inter-sentential transitions. The probability of the next sentence given the history is a coherence score.

---

## Conversation Acts and Dialogue Structure

Dialogue has an additional layer of structure: **conversational acts** (also called speech acts or dialogue moves). Austin's foundational work distinguished:

- **Constatives**: Asserting, claiming, informing (*The sky is blue*)
- **Performatives**: Actions performed by utterance (*I promise to come*)
- **Expressives**: Expressing psychological states (*I'm sorry*)

Modern dialogue act taxonomies (DAMSL, SWBD-DAMSI) include: statements, questions, directives, commutatives, agreements, and more.

Dialogue systems use these labels for intent classification and response generation:

```python
# Dialogue act classification
class DialogueActClassifier(nn.Module):
    def __init__(self, hidden_dim, num_acts):
        super().__init__()
        self.turn_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.act_classifier = nn.Linear(hidden_dim * 2, num_acts)

    def forward(self, utterance_tokens):
        # Encode the utterance
        hiddens, (hidden, _) = self.turn_encoder(utterance_tokens)
        # Use final hidden state for classification
        final_hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return self.act_classifier(final_hidden)

# Examples from SWBD-DAMSL:
labels = ['Statement', 'Question', 'Directive', 'Commissive', 'Expressive', 'Abridged']
# "Could you pass the salt?" -> Directive
# "I'll be there at 5." -> Commissive
```

---

## Key Concepts

- **Discourse analysis**: Study of how sentences form coherent larger texts
- **Rhetorical Structure Theory (RST)**: Hierarchical discourse structure with nucleus-satellite relations
- **Coreference resolution**: Identifying when expressions refer to the same entity
- **Entity grid model**: Matrix representation of entity grammatical roles for coherence scoring
- **Conversational acts / speech acts**: Pragmatic categories of utterances (assertives, directives, commissives, expressives)
- **Information flow**: The given-new contract in discourse; managing what information is new vs. familiar

## Exercises

1. **RST annotation**: Take a paragraph from a newspaper article and annotate it with RST relations. Identify the nuclei and satellites.
2. **Coreference resolution**: Find a short story (5-10 sentences) and manually annotate all coreference chains. What types of mentions (pronouns, names, definite NPs) are most common?
3. **Coherence scoring**: Compare the entity grid coherence score of a coherent paragraph vs. a randomly shuffled version of the same sentences. Is the difference significant?

## Further Reading

- Mann, W.C. & Thompson, S.A. (1988). "Rhetorical Structure Theory." *Text* 8(3).
- Barzilay, R. & Lapata, M. (2008). "Modeling Local Coherence." *Computational Linguistics* 34(1).
- Lee, K. et al. (2017). "End-to-end Neural Coreference Resolution." *EMNLP 2017*.
- Grosz, B.J. et al. (1995). "Centering: A Framework for Modeling Local Coherence." *Computational Linguistics* 21(2).
