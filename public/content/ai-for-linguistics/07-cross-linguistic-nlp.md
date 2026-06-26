---
title: "Cross-Linguistic NLP: Typology, Multilingual Models, and Code-Switching"
difficulty: intermediate
topic: ai-for-linguistics
order: 7
estimatedTime: "30 minutes"
summary: "Studies how linguistic typology affects NLP systems, how multilingual models like mBERT and XLM-R enable cross-lingual transfer, and the challenges posed by code-switching in multilingual communities."
---
# Cross-Linguistic NLP: Typology, Multilingual Models, and Code-Switching

## Overview

The world's approximately 7,000 languages differ from each other in striking ways. English has strict subject-verb-object word order (*The dog bit the man*); Japanese freely permutes constituents via case particles; Mandarin uses topic-comment structure; Arabic's non-concatenative morphology encodes grammatical information through vowel patterns within consonantal roots; Warlpiri has split ergativity based on tense.

This linguistic diversity is both a challenge and an opportunity for NLP. **Cross-linguistic NLP** studies how to build systems that work across many languages — and how the structural properties of languages affect what models learn. Multilingual models like mBERT and XLM-R are trained on 100+ languages simultaneously and show surprising cross-lingual transfer. But they also exhibit bias toward high-resource languages and the typological features encoded in their training data.

---

## Linguistic Typology

**Typology** is the study of how languages systematically vary. Key dimensions:

### Word Order Typology

| Order | Example | Languages |
|-------|---------|-----------|
| SVO | The dog bit the man | English, Mandarin, Spanish |
| SOV | The man the dog bit | Japanese, Korean, Hindi |
| VSO | Bit the dog the man | Classical Arabic, Irish |
| VOS | Bit the man the dog | Malagasy |
| OVS | The man the dog bit | Hiw (Vanuatu) |
| OSV | The man the dog bit | Apurímac (Peru) |

**Greenberg's universals**: Many word-order correlations are not accidental. If a language has prepositions, it likely has OV order (Japanese: *nihon ni* = Japan-DAT = "in Japan"; *ni* is a postposition-like case marker). These correlations emerge from processing efficiency trade-offs.

### Morphological Typology

Recall from Lesson 2: isolating (little morphology), agglutinating (transparent morphemes), fusional (multiple meanings per morpheme), introflexional (templatic). This affects how much a language relies on word order vs. morphology for encoding grammatical relations.

### Null Subject Typology

**Pro-drop languages** (Italian, Spanish, Japanese) allow subject pronouns to be omitted; **non-pro-drop languages** (English, French) require them. This correlates with rich verbal morphology:

$$\text{Italian: } \text{Canto} \underbrace{\emptyset}_{\text{1sg subject}} \implies \text{English: } \underbrace{I}_{\text{explicit subject}} \text{ sing}$$

---

## Multilingual Embeddings and Models

### Cross-lingual Word Embeddings

Cross-lingual word embeddings project words from different languages into a shared space, enabling transfer. Methods include:

- **Parallel data supervision**: Use word alignments from parallel corpora to learn a mapping $W$ such that $W \mathbf{u}_i \approx \mathbf{v}_i$ for aligned pairs $(u_i, v_i)$. The objective: $\min_W \sum_i \| W \mathbf{u}_i - \mathbf{v}_i \|^2$.

- **Self-learning**: Iteratively refine the mapping using the model's own predictions.

- **MUSE** (Conneau et al., 2018): Supervised and unsupervised cross-lingual word embedding methods.

```python
# Cross-lingual mapping: align word embedding spaces
def align_embeddings(embeddings_src, embeddings_tgt, vocab_src, vocab_tgt, n_iters=10):
    """
    Learn orthogonal transformation W to align source embeddings to target.
    Uses Procrustes analysis for supervised alignment.
    """
    from scipy.linalg import orthogonal_procrustes
    # Find common words to use as anchor points
    common = set(vocab_src.keys()) & set(vocab_tgt.keys())
    src_indices = [vocab_src[w] for w in common]
    tgt_indices = [vocab_tgt[w] for w in common]

    X = embeddings_src[src_indices]  # (N, d)
    Y = embeddings_tgt[tgt_indices]  # (N, d)

    # Procrustes: find W minimizing ||WX - Y||_F
    R, _ = orthogonal_procrustes(X, Y)
    return R

# After alignment: cosine similarity across languages measures semantic equivalence
# word_sim('dog', 'perro', aligned_model) should be high
```

### Multilingual BERT (mBERT)

mBERT is BERT trained on a concatenated corpus of 104 languages using shared vocabulary (WordPiece with language-specific vocabularies sharing the same indices). It is trained with MLM only — no cross-lingual signal.

Surprisingly, mBERT exhibits **zero-shot cross-lingual transfer**: a model fine-tuned on English NER data achieves reasonable NER performance on German without any German training data. This suggests the model learns language-agnostic structural representations.

But transfer is better for typologically similar languages and degrades for distant ones.

### XLM-RoBERTa (XLM-R)

XLM-R (Conneau et al., 2020) scales mBERT with 2.5TB of CommonCrawl data across 100 languages, using a larger model (base: 270M params, large: 560M params). Key finding: cross-lingual performance improves with scale, and a single unified model outperforms language-specific models for most languages.

---

## Code-Switching

**Code-switching** is the practice of alternating between two or more languages within a single conversation or utterance. It is extremely common in multilingual communities: Spanish-English in the US Southwest (*"Voy a go to the store"*), Arabic-French in Lebanon, Hindi-English in urban India.

Code-switching poses severe challenges for NLP:

1. **Language identification**: Which parts of the text are in which language? Standard tools fail.
2. **Tokenization**: SentencePiece/BPE trained on monolingual text misbehaves on mixed text.
3. **Morphological parsing**: Rules designed for one language don't apply.

```python
# Code-switched Spanish-English NER
# Input: "El presidente Obama visited Mexico yesterday"
# Expected: El(O) presidente(O) Obama(B-PER) visited(O) Mexico(B-LOC) yesterday(O)

# Challenges:
# 1. Language identification: "El" is Spanish, "presidente" is Spanish,
#    "Obama" is English (named entity), "Mexico" is Spanish (geographic)
# 2. Shared vocabulary: "visite d" vs "visited" - same root, different language
# 3. Named entities cross language boundaries

class CodeSwitchedTagger(nn.Module):
    """Sequence tagger that handles code-switching"""
    def __init__(self, vocab_size, num_langs, num_tags, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.lang_emb = nn.Embedding(num_langs, embed_dim // 4)
        self.word_emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm     = nn.LSTM(embed_dim + embed_dim // 4, hidden_dim, batch_first=True, bidirectional=True)
        self.tag_proj = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, word_ids, lang_ids):
        # lang_ids: dynamically predicted or given language of each token
        w_emb = self.word_emb(word_ids)
        l_emb = self.lang_emb(lang_ids)
        combined = torch.cat([w_emb, l_emb], dim=-1)
        hiddens, _ = self.lstm(combined)
        return self.tag_proj(hiddens)
```

---

## Typological Features and Model Performance

Recent work (LLMSurgeon, arXiv:2605.30348) shows that language models trained on mixed multilingual data develop language-specific "subnetworks" within their parameters. The distribution of languages in training data affects how well the model represents each language's typological features.

Key findings:
- **Morphology-rich languages** benefit from more parameters (more capacity to encode complex inflection)
- **Word-order自由度** (free word order) is harder to capture with attention mechanisms optimized for SVO
- **Script similarity** (Latin-script languages share representations even if genealogically distant)
- **Data quality matters more than quantity** for low-resource languages

---

## Key Concepts

- **Typological variation**: Systematic cross-linguistic differences in word order, morphology, and syntax
- **Greenberg's universals**: Correlations among typological features (e.g., prepositions correlate with OV order)
- **Cross-lingual transfer**: Using a model trained on one language to perform tasks in another
- **mBERT / XLM-R**: Multilingual transformers trained on 100+ languages with shared vocabulary
- **Zero-shot transfer**: Performing a task in a language with no training data, using representations from a multilingual model
- **Code-switching**: Alternating between languages in a single utterance; a major challenge for NLP
- **Pro-drop**: Languages that allow (or require) omission of subject pronouns

## Exercises

1. **Typological analysis**: Choose two languages from different families (e.g., Turkish and English). Compare their morphosyntactic properties: word order, case system, agreement, and pro-drop. How would these differences affect an NLP pipeline for each?
2. **Cross-lingual transfer**: Fine-tune mBERT on English POS tagging. Evaluate zero-shot on German. Compare with a German-specific model. How large is the gap?
3. **Code-switching**: Find examples of code-switching in social media text (Twitter, WhatsApp). What linguistic phenomena make automatic processing difficult?

## Further Reading

- Conneau, A. et al. (2018). "Word Translation Without Parallel Data." *ICLR*.
- Conneau, A. et al. (2020). "Unsupervised Cross-Lingual Representation Learning at Scale." *ACL*.
- Pires, T. et al. (2019). "How Multilingual is Multilingual BERT?" *ACL*.
- "LLMSurgeon: Diagnosing Data Mixture of Large Language Models" (arXiv:2605.30348, ACL 2026).
- Ahuja, O. et al. (2023). "The Grammar of Populations in the Brain." *Nature Human Behaviour* (on typological biases in models).
