---
title: "Computational Morphology: Morphological Analysis and Generation"
difficulty: beginner
topic: ai-for-linguistics
order: 2
estimatedTime: "30 minutes"
summary: "Covers computational approaches to morphology, from finite-state transducers for morpheme segmentation and generation to neural sequence-to-sequence models for morphological inflection."
---

# Computational Morphology: Morphological Analysis and Generation

## Overview

Morphology — the study of word formation — is the interface between vocabulary and grammar. Every language has a system of meaningful units (morphemes) that combine to create words: English *-s* marks plural, *un-* signals negation, *-ly* converts adjectives to adverbs. These morphemes are the smallest meaning-bearing units, and how to segment, analyze, and generate them computationally is the subject of **computational morphology**.

For NLP systems, morphological processing is foundational. Tokenizing text into words is only the first step; understanding that *unkindest* consists of *un-* + *kind* + *est* enables better parsing, semantic interpretation, and machine translation.Morphological knowledge also improves performance in low-resource settings: a model that understands *walked* = walk + PAST can generalize from *jumped* = jump + PAST even without seeing *jumped* in training data.

---

## Morphological Types

Languages differ dramatically in how they build words:

**Isolating languages** (e.g., Mandarin Chinese) have very little inflectional morphology; most words are monomorphemic. The grammatical relationship is expressed through word order.

**Agglutinating languages** (e.g., Turkish, Finnish, Japanese) string multiple morphemes together in a relatively transparent way: Turkish *geliyorum* = gel (come) + -iyor (progressive) + -um (1st person singular). Each morpheme corresponds to one grammatical meaning.

**Fusional languages** (e.g., Spanish, Russian, Latin) combine multiple grammatical meanings into single morphemes: Spanish *canté* = cant (sing) + -a (past tense) + -é (1st person singular). A single ending encodes tense, person, and number.

**Introflexional languages** (e.g., Arabic) encode grammatical information through internal vowel changes (non-concatenative morphology): Arabic *kataba* (he wrote) vs. *yuktabu* (it is written).

English has both concatenative morphology (prefixes and suffixes) and some non-concatenative patterns (sing/sang/sung).

---

## Finite-State Methods

The classical computational approach to morphology uses finite-state transducers (FSTs), which model the mapping between surface forms and lexical representations.

An FST is a tuple $(Q, \Sigma, \Delta, q_0, F, \delta)$ where:
- $Q$ is a finite set of states
- $\Sigma$ is the input alphabet (characters or morphemes)
- $\Delta$ is the output alphabet
- $q_0 \in Q$ is the start state
- $F \subseteq Q$ is the set of final states
- $\delta: Q \times \Sigma \rightarrow Q$ is the transition function

Two FSTs can be composed: a **lexicographer** maps surface forms to morpheme sequences, and a **generator** maps morpheme sequences to surface forms.

```python
# Finite-state morphology simulation using Python
# This demonstrates the two-level morphology concept

class FST:
    def __init__(self, states, alphabet, transitions, start, finals):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # dict: (state, input) -> (next_state, output)
        self.start = start
        self.finals = finals

    def analyze(self, input_string):
        """Surface form -> morphological parse"""
        state = self.start
        output = []
        i = 0
        while i < len(input_string):
            key = (state, input_string[i])
            if key in self.transitions:
                state, out_sym = self.transitions[key]
                output.append(out_sym)
                i += 1
            else:
                return None  # No valid transition
        if state in self.finals:
            return '/'.join(output)
        return None

    def generate(self, parse):
        """Morphological parse -> surface form"""
        state = self.start
        surface = []
        for sym in parse.split('/'):
            found = False
            for (s, inp), (ns, out) in self.transitions.items():
                if s == state and out == sym:
                    state = ns
                    surface.append(inp)
                    found = True
                    break
            if not found:
                return None
        if state in self.finals:
            return ''.join(surface)
        return None

# A tiny English noun FST: singular <-> plural
noun_fst = FST(
    states={'q0', 'q1', 'q2'},
    alphabet='abcdefghijklmnopqrstuvwxyz',
    transitions={
        ('q0', 'c'): ('q1', 'c'),   # consonant cluster (stem)
        ('q1', 'V'): ('q1', 'V'),   # vowels within stem
        ('q1', 'y'): ('q1', 'y'),
        ('q1', 's'): ('q2', 's'),   # plural suffix
        ('q2', 'e'): ('q2', 'e'),   # -es allomorph
    },
    start='q0',
    finals={'q1', 'q2'}
)

# Test analyses
test_words = ['cats', 'dogs', 'buses', 'flies']
for word in test_words:
    parse = noun_fst.analyze(word)
    print(f"{word:10} -> {parse}")
```

---

## Neural Approaches to Morphology

Modern systems often use neural sequence labeling for morphological tasks:

**Morpheme segmentation**: Given a word like *unbreakable*, output *un/break/able*. Models range from LSTMs with CRF layers to transformers. Character-level models naturally handle unseen words and non-concatenative morphology.

**Morphological tagging**: Assign each morpheme a tag (NEG, PAST, PL, etc.). The input is a segmented or unsegmented word, and the output is a tag sequence.

**Inflection generation**: Given a lemma (e.g., *go*) and a target tag (e.g., PAST, 1SG), generate the inflected form (*went*). This is a seq2seq task that modern models handle well, even for languages with rich morphology.

```python
# Character-level RNN for morphological inflection
# Simplified pseudocode — in practice you'd use a proper framework

import torch
import torch.nn as nn

class CharMorphInflector(nn.Module):
    """
    Sequence-to-sequence model for morphological inflection.
    Input: character sequence of lemma + target morphological tags
    Output: character sequence of inflected form
    """
    def __init__(self, char_vocab_size, tag_vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, embed_dim)
        self.tag_embed = nn.Embedding(tag_vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, char_vocab_size)

    def forward(self, lemma_chars, tags):
        # Encode lemma characters
        lemma_emb = self.char_embed(lemma_chars)          # [B, L, E]
        tag_emb   = self.tag_embed(tags).expand(-1, lemma_emb.size(1), -1)  # [B, L, E]
        encoded   = self.encoder(lemma_emb)               # [B, L, 2H]

        # Decode to produce inflected form
        decoder_input = torch.zeros_like(encoded[:, 0:1, :])  # start token
        outputs = []
        for t in range(20):  # max generation length
            rnn_out = self.decoder(decoder_input)          # [B, 1, H]
            logits  = self.output_proj(rnn_out)             # [B, 1, V]
            outputs.append(logits)
            # (in practice: use teacher forcing, attention, and proper EOS handling)
        return torch.cat(outputs, dim=1)

# Example: go + PAST -> went
# The model must learn the vowel change (ablaut) pattern
```

---

## Key Concepts

- **Morpheme**: The smallest meaningful unit in a language (root, prefix, suffix, infix)
- **Allomorph**: Variant forms of a morpheme (English plural -s/-es/-en as in *oxen*)
- **Finite-state transducer (FST)**: A formalism for mapping between surface forms and morphological analyses
- **Inflection vs. derivation**: Inflection (walk → walks) is grammatically obligatory; derivation (walk → walkable) changes word class or meaning
- **Non-concatenative morphology**: Morphology via templatic or reduplicative processes (Arabic, Semitic languages)
- **Seq2seq inflection**: Casting morphological generation as encoder-decoder sequence transduction

## Exercises

1. **Morpheme segmentation**: Segment these words into morphemes and identify each morpheme's category: *unfriendly*, *prehistoric*, *reorganization*, *misunderstandings*, *irrefutability*.
2. **Language comparison**: Compare the morphological typology of English, Turkish, and Japanese. How would each language express "the cats were sleeping"?
3. **FST design**: Design an FST that maps between the imperative forms *walk*, *walk-s* (archaic second-person singular) and their morphological analyses.

## Further Reading

- Beesley, K.R. & Karttunen, L. (2003). *Finite State Morphology*. CSLI Publications.
- Cotterell, R. et al. (2018). "A Neural Language Model for Morphologically Rich Languages." *Transactions of the ACL*.
- Kann, S. et al. (2020). "The SIGMORPHON 2020 Shared Task on Morphological Analysis." *ACL 2020 Proceedings*.
