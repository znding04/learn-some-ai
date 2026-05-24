---
title: "Diffusion Models for Audio Deep Dive"
level: intermediate
topic: ai-for-music-and-creative-arts
order: 6
---

# Diffusion Models for Audio Deep Dive

## Overview

Diffusion models have become one of the two dominant paradigms for high-quality audio generation, alongside transformer-based approaches. While Lesson 3 introduced the basic concept of diffusion — iteratively denoising random noise into structured audio — this lesson dives deeper into the specific architectures, training techniques, and conditioning mechanisms that make diffusion models excel at music generation.

The breakthrough system in this space is **Stable Audio** by Stability AI. Stable Audio applies latent diffusion (the same framework behind Stable Diffusion for images) to audio generation, operating in a compressed latent space learned by a variational autoencoder. This approach generates high-fidelity, stereo, 44.1 kHz audio up to 6 minutes long — a significant leap from earlier systems that struggled with clips beyond 30 seconds.

What makes diffusion models particularly attractive for music is their natural support for **controllable generation**. Through classifier-free guidance, you can steer the denoising process toward specific genres, moods, or instruments. Through inpainting, you can regenerate specific sections of audio while keeping the rest intact. And through interpolation in the noise space, you can create smooth transitions between musical styles. These capabilities make diffusion models powerful tools for both automated generation and interactive creative workflows.

---

## Denoising Diffusion in the Audio Domain

### The Forward and Reverse Process

The forward diffusion process adds noise according to a variance schedule β_1, ..., β_T:

```
q(x_t | x_{t-1}) = N(x_t; √(1 - β_t) · x_{t-1}, β_t · I)
```

A useful property allows jumping directly to any timestep t:

```
q(x_t | x_0) = N(x_t; √(ᾱ_t) · x_0, (1 - ᾱ_t) · I)

where ᾱ_t = ∏_{s=1}^{t} (1 - β_s)
```

The reverse process uses a neural network to predict the noise:

```python
import torch
import torch.nn as nn

class AudioDiffusionModel(nn.Module):
    """U-Net style model for audio diffusion (simplified)."""
    
    def __init__(self, latent_dim=64, time_dim=256, cond_dim=512):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Conditioning projection (text embeddings)
        self.cond_proj = nn.Linear(cond_dim, time_dim)
        
        # U-Net encoder
        self.down1 = ResBlock(latent_dim, 128, time_dim)
        self.down2 = ResBlock(128, 256, time_dim)
        self.down3 = ResBlock(256, 512, time_dim)
        
        # Bottleneck with cross-attention to text
        self.mid_attn = CrossAttention(512, cond_dim)
        self.mid_block = ResBlock(512, 512, time_dim)
        
        # U-Net decoder (with skip connections)
        self.up3 = ResBlock(1024, 256, time_dim)  # 512 + 512 skip
        self.up2 = ResBlock(512, 128, time_dim)    # 256 + 256 skip
        self.up1 = ResBlock(256, latent_dim, time_dim)
        
        self.out = nn.Conv1d(latent_dim, latent_dim, 1)
    
    def forward(self, x_t, t, text_cond):
        t_emb = self.time_mlp(t) + self.cond_proj(text_cond.mean(dim=1))
        
        # Encoder
        h1 = self.down1(x_t, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.down3(h2, t_emb)
        
        # Bottleneck
        h = self.mid_attn(h3, text_cond)
        h = self.mid_block(h, t_emb)
        
        # Decoder with skip connections
        h = self.up3(torch.cat([h, h3], dim=1), t_emb)
        h = self.up2(torch.cat([h, h2], dim=1), t_emb)
        h = self.up1(torch.cat([h, h1], dim=1), t_emb)
        
        return self.out(h)  # predicted noise
```

---

## Stable Audio Architecture: Latent Diffusion

Stable Audio's architecture consists of three components:

```
┌─────────────────────────────────────────────────────┐
│                 Stable Audio Pipeline                 │
│                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ VAE      │    │ Diffusion    │    │ VAE        │ │
│  │ Encoder  │───→│ U-Net        │───→│ Decoder    │ │
│  │          │    │ (in latent)  │    │            │ │
│  └──────────┘    └──────┬───────┘    └────────────┘ │
│                         │                             │
│              ┌──────────┴──────────┐                 │
│              │ Text Encoder (CLAP) │                 │
│              │ + Timing Embeddings │                 │
│              └─────────────────────┘                 │
└─────────────────────────────────────────────────────┘
```

### The VAE (Autoencoder)

The VAE compresses raw audio (44.1 kHz stereo) into a latent space with a compression ratio of ~2048x:

- Input: 44,100 samples/sec × 2 channels = 88,200 values/sec
- Latent: ~43 latent frames/sec × 64 channels
- This makes the diffusion process computationally feasible

### Timing Conditioning

A unique feature of Stable Audio is **timing conditioning** — the model is conditioned on the total duration and start/end timestamps:

```python
# Timing conditioning allows generating specific durations
# and controlling where musical events occur
timing_cond = {
    "seconds_start": 0.0,    # Start of the clip
    "seconds_total": 180.0,  # Total song duration (3 min)
}
# The model learns to generate audio appropriate for this position
# in a longer piece — e.g., intro vs. chorus vs. outro
```

---

## Classifier-Free Guidance for Audio

Classifier-free guidance (CFG) is the primary mechanism for controlling how closely generated audio follows the text prompt:

### Training

During training, the text condition is randomly dropped (replaced with a null embedding) with some probability (typically 10-20%):

```python
def training_step(model, x_0, text_cond, drop_prob=0.1):
    # Randomly drop conditioning
    mask = torch.rand(x_0.shape[0]) < drop_prob
    cond = text_cond.clone()
    cond[mask] = null_embedding  # empty/unconditional
    
    # Standard diffusion training
    t = torch.randint(0, T, (x_0.shape[0],))
    noise = torch.randn_like(x_0)
    x_t = add_noise(x_0, noise, t)
    predicted_noise = model(x_t, t, cond)
    loss = F.mse_loss(predicted_noise, noise)
    return loss
```

### Inference

At inference time, the model runs twice per denoising step — once with and once without the text condition:

```python
def guided_denoise_step(model, x_t, t, text_cond, guidance_scale=3.5):
    # Unconditional prediction
    noise_uncond = model(x_t, t, null_embedding)
    # Conditional prediction
    noise_cond = model(x_t, t, text_cond)
    # Guided prediction: amplify the difference
    noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    return noise_guided
```

**Guidance scale effects on music:**
- **1.0**: Unconditional — ignores the prompt, generates generic music
- **2.0-3.0**: Balanced — follows the prompt while maintaining natural variation
- **4.0-5.0**: Strong guidance — closely matches the prompt but may sound over-processed
- **7.0+**: Very strong — can produce artifacts and reduced quality

---

## Music-Specific Conditioning

### Genre and Mood

Text encoders like CLAP (Contrastive Language-Audio Pretraining) are trained to align text descriptions with audio features:

```python
# CLAP encodes both text and audio into a shared embedding space
from msclap import CLAP

clap = CLAP(version="2023", use_cuda=True)

# These text embeddings guide the diffusion process
text_emb = clap.get_text_embeddings(["dark ambient electronic music with deep bass drones"])
# The resulting embedding encodes genre, mood, and instrumentation in a form
# that the diffusion U-Net can use for cross-attention conditioning
```

### Tempo and Rhythm Control

While exact BPM control remains challenging, tempo can be influenced through:
- Text prompts: "slow waltz at 90 BPM", "fast techno at 140 BPM"
- Beat conditioning: providing a reference rhythmic pattern
- Post-processing: time-stretching generated audio to match a target tempo

---

## Key Concepts

- **Latent Diffusion**: Running the denoising process in a compressed latent space rather than on raw audio waveforms. Reduces computation by orders of magnitude.
- **U-Net Architecture**: An encoder-decoder with skip connections that processes the noisy latent at multiple resolutions. Cross-attention layers inject text conditioning.
- **Classifier-Free Guidance (CFG)**: Training with random conditioning dropout, then amplifying the difference between conditioned and unconditioned predictions at inference time.
- **Timing Conditioning**: Providing the model with temporal position information so it can generate music appropriate for a specific point in a longer piece.
- **CLAP (Contrastive Language-Audio Pretraining)**: A model that learns shared embeddings between text descriptions and audio, enabling text-conditioned generation.

---

## Further Reading

- Evans et al., "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" (Stability AI, 2024)
- Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)
- Wu et al., "Large-Scale Contrastive Language-Audio Pretraining" (CLAP, 2023)
- Song et al., "Denoising Diffusion Implicit Models" (DDIM — faster sampling, 2020)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)
