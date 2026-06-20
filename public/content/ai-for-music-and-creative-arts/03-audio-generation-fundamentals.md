---
title: "Audio Generation Fundamentals"
difficulty: beginner
topic: ai-for-music-and-creative-arts
order: 3
estimatedTime: "30 minutes"
summary: "Introduces the core approaches to neural audio synthesis, including autoregressive models like WaveNet, diffusion models for audio, latent diffusion, and vocoders like HiFi-GAN and Encodec."
---

## Audio Generation Fundamentals

## Overview

Generating audio with neural networks is fundamentally a sequence generation problem — but one with extreme demands. A single second of CD-quality audio contains 44,100 floating-point values. A 3-minute song requires generating over 7.9 million values that must be temporally coherent at both the microsecond level (individual waveform cycles) and the multi-second level (musical phrases and song structure). This multi-scale coherence requirement makes audio generation one of the most challenging domains in generative AI.

The field has evolved through several generations of approaches. Early neural audio synthesis used autoregressive models like WaveNet (2016), which generated audio one sample at a time — producing stunning quality but requiring hours to generate a single second of audio. The field then moved to spectrogram-based approaches: generate a mel spectrogram with one model, then convert it to audio with a vocoder. More recently, diffusion models operating in learned latent spaces have emerged as the dominant paradigm, offering both high quality and practical generation speeds.

Understanding these fundamentals — how raw waveforms are synthesized, how diffusion models denoise random noise into structured audio, and how vocoders bridge the gap between abstract representations and audible sound — provides the foundation for understanding every modern music generation system.

---

## Waveform Synthesis and Sampling

### The Autoregressive Approach: WaveNet

WaveNet, introduced by DeepMind in 2016, modeled audio as an autoregressive process. Each sample is predicted based on all previous samples using dilated causal convolutions:

```text
Sample prediction:
x(t) = f(x(t-1), x(t-2), ..., x(t-R))

Dilated convolutions expand the receptive field:
Layer 1: dilation = 1   [looks back 1 sample]
Layer 2: dilation = 2   [looks back 2 samples]
Layer 3: dilation = 4   [looks back 4 samples]
...
Layer 10: dilation = 512 [looks back 512 samples]
```

```python
import torch
import torch.nn as nn

class DilatedCausalConv(nn.Module):
    """A single dilated causal convolution layer (simplified WaveNet block)."""
    def __init__(self, channels, kernel_size=2, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation,
            padding=dilation  # causal padding
        )
        self.gate = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation,
            padding=dilation
        )

    def forward(self, x):
        # Gated activation (key to WaveNet's expressiveness)
        h = torch.tanh(self.conv(x)) * torch.sigmoid(self.gate(x))
        return x + h  # residual connection
```

WaveNet produced remarkable audio quality but was impractically slow for real-time use — generating one sample at a time at 44,100 Hz meant generating ~44,000 sequential neural network forward passes per second.

### Parallel Waveform Generation

Subsequent work focused on parallel generation:

- **Parallel WaveNet** (2017): Used a trained WaveNet as a teacher to train a fast parallel student model via probability density distillation
- **WaveGlow** (2018): Flow-based model generating all samples simultaneously
- **HiFi-GAN** (2020): GAN-based vocoder achieving real-time synthesis with quality matching autoregressive models

---

## Diffusion Models for Audio

### Denoising Diffusion Probabilistic Models (DDPMs)

Diffusion models generate audio by learning to reverse a noise-adding process. The forward process gradually adds Gaussian noise to audio until it becomes pure noise. The model learns to reverse each step:

```text
Forward process (adding noise):
x_0 (clean audio) → x_1 → x_2 → ... → x_T (pure noise)

Reverse process (denoising):
x_T (noise) → x_{T-1} → ... → x_1 → x_0 (generated audio)
```

The mathematical formulation:

**Forward process**: q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)

**Reverse process**: p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² · I)

The neural network learns to predict the noise ε that was added at each timestep:

```python
import torch
import torch.nn as nn

class SimpleDiffusionStep(nn.Module):
    """Simplified noise prediction network for audio diffusion."""
    def __init__(self, audio_dim, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(audio_dim + time_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, audio_dim)
        )

    def forward(self, noisy_audio, timestep):
        t_emb = self.time_embed(timestep.float().unsqueeze(-1))
        x = torch.cat([noisy_audio, t_emb], dim=-1)
        predicted_noise = self.net(x)
        return predicted_noise

# Training loop (conceptual)
def train_step(model, clean_audio, noise_schedule):
    t = torch.randint(0, len(noise_schedule), (clean_audio.shape[0],))
    noise = torch.randn_like(clean_audio)
    noisy = noise_schedule.add_noise(clean_audio, noise, t)
    predicted = model(noisy, t)
    loss = nn.MSELoss()(predicted, noise)
    return loss
```

### Latent Diffusion for Audio

Running diffusion directly on raw audio waveforms is computationally expensive. **Latent diffusion** (used by Stable Audio) first compresses audio into a lower-dimensional latent space using a VAE, runs diffusion there, then decodes:

```text
Audio (44100 Hz) → VAE Encoder → Latent (compressed) → Diffusion → Decoded Latent → VAE Decoder → Audio
```

This reduces the sequence length by 100-500x while preserving perceptual quality.

---

## Vocoders: From Spectrograms to Sound

A vocoder converts mel spectrograms (or other intermediate representations) into audible waveforms. Modern vocoders are critical components in most music generation pipelines.

### HiFi-GAN

HiFi-GAN uses a generator with multi-receptive-field fusion and multiple discriminators operating at different scales:

```text
Mel Spectrogram → Transposed Convolutions → Multi-Receptive-Field Fusion → Waveform
                                                    ↕
                                    Multi-Scale Discriminator (adversarial training)
                                    Multi-Period Discriminator (captures periodic structure)
```

HiFi-GAN achieves near-perfect perceptual quality while running faster than real-time on a GPU.

### Encodec as a Vocoder

Encodec's decoder serves as a neural vocoder that converts discrete token sequences back to audio. Unlike mel-spectrogram vocoders, it operates on learned discrete codes:

```text
Token IDs → Codebook Lookup → Sum Quantized Vectors → Decoder (ConvTranspose1d layers) → Waveform
```

---

## Key Concepts

- **Autoregressive Generation**: Generating audio one sample (or token) at a time, conditioned on all previous outputs. High quality but slow.
- **Diffusion Models**: Generate by iteratively denoising random noise. Can generate all timesteps in parallel within each denoising step.
- **Latent Diffusion**: Performing diffusion in a compressed space rather than directly on audio, reducing computation by orders of magnitude.
- **Vocoder**: Converts intermediate representations to audible waveforms. Modern vocoders (HiFi-GAN, Encodec decoder) run in real-time.
- **Noise Schedule**: The sequence of noise levels used during diffusion training and sampling. Controls the trade-off between diversity and quality.

---

## Further Reading

- van den Oord et al., "WaveNet: A Generative Model for Raw Audio" (DeepMind, 2016)
- Kong et al., "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020)
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)
- Evans et al., "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" (2024)
