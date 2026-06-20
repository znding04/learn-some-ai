---
title: "Transformers for Music Generation"
difficulty: beginner
topic: ai-for-music-and-creative-arts
order: 4
estimatedTime: "30 minutes"
summary: "Explains how transformer architectures are applied to music generation, covering MusicLM's hierarchical approach, MusicGen's codebook delay pattern, self-attention over music sequences, and conditioning mechanisms."
---

## Transformers for Music Generation

## Overview

Transformers have become the dominant architecture for music generation, just as they have for text and image generation. The key insight enabling this was treating music as a sequence prediction problem: if audio can be represented as a sequence of discrete tokens (via neural codecs like Encodec), then the same autoregressive transformer architecture that powers large language models can be applied to generate music. The model learns to predict the next audio token given all previous tokens — and through this simple objective, it captures melody, harmony, rhythm, instrumentation, and even high-level song structure.

Two landmark systems demonstrated this approach: Google's **MusicLM** (2023) introduced a hierarchical framework where text descriptions are first mapped to semantic audio tokens and then to acoustic tokens. Meta's **MusicGen** (2023) simplified this into a single-stage transformer operating over multiple codebook streams simultaneously. Both systems accept text conditioning (e.g., "mellow jazz piano with brushed drums") and produce coherent, high-quality music lasting 10-30 seconds.

The transformer approach to music generation is compelling because it leverages decades of research on scaling transformer models. Techniques developed for LLMs — attention mechanisms, positional encodings, KV caching for efficient inference, classifier-free guidance for controllable generation — transfer directly to the music domain. This has enabled rapid progress: in just two years, we went from 5-second clips of moderate quality to full 6-minute songs with vocals, multiple instruments, and recognizable song structure.

---

## MusicLM: Hierarchical Text-to-Music

MusicLM (Google, 2023) generates music from text through a cascade of three models:

```text
Text Description
      ↓
[MuLan Text Encoder] → Text embedding
      ↓
[Semantic Modeling Stage] → Semantic tokens (from w2v-BERT)
      ↓
[Acoustic Modeling Stage] → Acoustic tokens (from SoundStream)
      ↓
[SoundStream Decoder] → Audio waveform
```

Each stage is a transformer that predicts tokens conditioned on the output of the previous stage:

1. **Semantic Stage**: Maps text embeddings to semantic audio tokens that capture high-level musical content (melody, rhythm, genre) without fine acoustic detail.

2. **Acoustic Stage**: Converts semantic tokens to acoustic tokens that encode the full spectral detail needed for high-fidelity audio.

This hierarchical approach allows each stage to focus on a different level of abstraction, similar to how image generation models separate layout/composition from fine pixel details.

---

## MusicGen: Single-Stage Multi-Codebook Generation

Meta's MusicGen simplified the pipeline to a single autoregressive transformer that generates multiple codebook streams:

```python
# Conceptual MusicGen token generation
# Audio is encoded into K codebooks, each with T timesteps
# Codebook 1: coarse (melody, rhythm)
# Codebook 2-4: progressively finer detail

# Token layout (delay pattern):
# Time:     t=0   t=1   t=2   t=3   t=4
# Book 1:   a0    a1    a2    a3    a4
# Book 2:   -     b0    b1    b2    b3
# Book 3:   -     -     c0    c1    c2
# Book 4:   -     -     -     d0    d1

# The delay pattern allows parallel generation across codebooks
# while maintaining autoregressive ordering
```

The key innovation is the **codebook interleaving pattern**. Rather than generating all codebooks for one timestep before moving to the next (which would be slow), MusicGen uses a delay pattern that shifts each codebook by one position. This allows the model to generate one token per forward pass while still producing all codebook levels:

```python
from audiocraft.models import MusicGen

# Load model — available sizes: small (300M), medium (1.5B), large (3.3B)
model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(
    duration=15,            # seconds
    top_k=250,              # top-k sampling for diversity
    top_p=0.0,              # disabled when top_k is set
    temperature=1.0,        # sampling temperature
    cfg_coef=3.0,           # classifier-free guidance strength
)

# Text-conditioned generation
descriptions = ["cinematic orchestral piece with rising strings and timpani"]
wav = model.generate(descriptions)  # shape: (1, 1, samples)

# Melody-conditioned generation (provide a reference melody)
import torchaudio
melody, sr = torchaudio.load("reference_melody.wav")
wav = model.generate_with_chroma(descriptions, melody, sr)
```

---

## Self-Attention Over Music Sequences

The self-attention mechanism is particularly well-suited to music because musical structure involves long-range dependencies:

- A chord progression established in bar 1 should be maintained or varied in bar 16
- A chorus melody introduced at 0:30 should recur at 1:30
- Rhythmic patterns repeat across measures but with variations

Standard self-attention computes attention between all pairs of tokens:

```text
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

where:
  Q = query matrix (what am I looking for?)
  K = key matrix (what do I contain?)
  V = value matrix (what information do I provide?)
  d_k = dimension of keys (scaling factor)
```

For a 15-second clip at 75 tokens/second with 4 codebooks, the sequence length is ~4,500 tokens. Self-attention over this length is feasible with modern GPU memory but becomes challenging for longer durations, which is why efficient attention variants (Flash Attention, sliding window attention) are important for music generation.

---

## Conditioning and Control Signals

Transformers for music accept various conditioning signals:

### Text Conditioning
Text descriptions are encoded using a pre-trained text encoder (T5, CLAP, or MuLan) and injected via cross-attention:

```python
# Cross-attention in a transformer decoder block
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, text_embeddings):
        # Self-attention over audio tokens
        x = x + self.self_attn(x, x, x)[0]
        # Cross-attention to text conditioning
        x = x + self.cross_attn(x, text_embeddings, text_embeddings)[0]
        # Feed-forward
        x = x + self.ffn(x)
        return x
```

### Melody Conditioning
MusicGen supports melody conditioning via chroma features — a 12-dimensional representation of pitch class energy extracted from a reference audio:

```text
Chroma vector at time t: [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
Values represent energy in each pitch class.
Preserves melody contour while allowing the model to choose timbre and arrangement.
```

### Classifier-Free Guidance (CFG)
CFG improves the alignment between text prompts and generated audio by interpolating between conditioned and unconditioned predictions:

```text
output = unconditioned + guidance_scale × (conditioned - unconditioned)
```

Higher guidance scales produce output more faithful to the prompt but with less diversity.

---

## Key Concepts

- **Codebook Delay Pattern**: An interleaving strategy that enables efficient multi-codebook generation with a single autoregressive pass.
- **Cross-Attention Conditioning**: Using attention between audio tokens and text/melody embeddings to steer generation.
- **Hierarchical Generation**: Breaking the text-to-audio problem into stages operating at different levels of abstraction.
- **Classifier-Free Guidance**: A training and inference technique that improves prompt adherence by comparing conditioned and unconditioned outputs.

---

## Further Reading

- Copet et al., "Simple and Controllable Music Generation" (MusicGen, Meta, 2023)
- Agostinelli et al., "MusicLM: Generating Music From Text" (Google, 2023)
- Vaswani et al., "Attention Is All You Need" (2017) — the foundational transformer paper
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
