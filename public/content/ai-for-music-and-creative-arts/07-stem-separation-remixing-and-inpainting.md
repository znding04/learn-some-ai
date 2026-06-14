---
title: "Stem Separation, Remixing, and Inpainting"
difficulty: intermediate
topic: ai-for-music-and-creative-arts
order: 7
estimatedTime: "30 minutes"
summary: "Covers AI tools for working with existing audio, including source separation with Demucs and Spleeter, music inpainting with diffusion models, and cross-modal editing techniques like style transfer and tempo modification."
---

# Stem Separation, Remixing, and Inpainting

## Overview

While generating music from scratch is impressive, some of the most practical AI music tools work with *existing* audio. Stem separation isolates individual instruments from a mixed recording — extracting the vocals from a pop song, or isolating the drum track from a full mix. Music inpainting fills in missing sections of audio, allowing you to extend a song, replace a flawed section, or smoothly transition between two clips. And AI-powered remixing lets you change the style, tempo, or instrumentation of existing music.

These capabilities have transformed music production. DJs use stem separation to create custom remixes. Producers use inpainting to seamlessly loop or extend sections. Film composers use style transfer to quickly prototype soundtrack variations. And music educators use source separation to create practice-along tracks with specific instruments removed.

The technical foundation for these capabilities is **source separation** — the "cocktail party problem" of isolating individual sound sources from a mixture. Deep learning has made this problem tractable: models like Meta's **Demucs** and Deezer's **Spleeter** can separate a mixed recording into vocals, drums, bass, and other instruments with remarkable quality. Combined with generative models for inpainting and style transfer, these tools form a complete AI-powered audio editing toolkit.

---

## Source Separation with Deep Learning

### The Problem

Given a mixed audio signal:

```text
mixture(t) = vocals(t) + drums(t) + bass(t) + other(t)
```

The goal is to recover each source signal. This is an underdetermined problem — there are infinitely many combinations of sources that could produce the same mixture. Deep learning models learn statistical priors about what each instrument "should" sound like.

### Demucs (Meta)

Demucs is the state-of-the-art open-source model for music source separation. It uses a hybrid architecture combining a temporal convolutional network with a spectrogram-based transformer:

```python
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio

# Load the pre-trained Hybrid Transformer Demucs model
model = get_model("htdemucs")
model.eval()

# Load a mixed audio file
mix, sr = torchaudio.load("song.wav")
# Resample to model's expected sample rate if needed
if sr != model.samplerate:
    mix = torchaudio.functional.resample(mix, sr, model.samplerate)

# Separate into stems
# Output shape: (sources, channels, samples)
# Sources: drums, bass, other, vocals
with torch.no_grad():
    sources = apply_model(model, mix.unsqueeze(0))[0]

# Save individual stems
stem_names = ["drums", "bass", "other", "vocals"]
for i, name in enumerate(stem_names):
    torchaudio.save(f"{name}.wav", sources[i], model.samplerate)
    print(f"Saved {name}.wav")
```

### Demucs Architecture

```text
Input Mix (stereo waveform)
        │
        ├──────────────────────────┐
        ▼                          ▼
┌───────────────┐         ┌───────────────┐
│ Temporal       │         │ Spectral      │
│ Encoder        │         │ Encoder       │
│ (Conv1d layers)│         │ (STFT + Conv) │
└───────┬───────┘         └───────┬───────┘
        │                          │
        ▼                          ▼
┌───────────────────────────────────────┐
│     Cross-Domain Transformer          │
│  (attends across time & frequency)    │
└───────────────┬───────────────────────┘
        │                          │
        ▼                          ▼
┌───────────────┐         ┌───────────────┐
│ Temporal       │         │ Spectral      │
│ Decoder        │         │ Decoder       │
└───────┬───────┘         └───────┬───────┘
        │                          │
        └──────────┬───────────────┘
                   ▼
        4 Separated Stems
   (drums, bass, other, vocals)
```

### Spleeter (Deezer)

Spleeter is an earlier, lighter-weight alternative using U-Net architectures on spectrograms:

```python
from spleeter.separator import Separator

# 2-stem (vocals + accompaniment), 4-stem, or 5-stem separation
separator = Separator("spleeter:4stems")
separator.separate_to_file("song.mp3", "output_directory/")
# Creates: vocals.wav, drums.wav, bass.wav, other.wav
```

---

## Music Inpainting

Music inpainting generates audio to fill masked regions, analogous to image inpainting. Use cases include:

- **Gap filling**: Repairing corrupted or missing sections
- **Extension**: Adding new bars at the end of a clip
- **Transition creation**: Generating smooth bridges between two sections

### Inpainting with Diffusion Models

Diffusion models naturally support inpainting. During the reverse process, known regions are replaced with the noised version of the original audio at each step, while masked regions are freely generated:

```python
def inpaint_step(model, x_t, t, original_audio, mask, text_cond):
    """
    mask: binary tensor (1 = generate, 0 = keep original)
    """
    # Denoise the full sequence
    predicted_noise = model(x_t, t, text_cond)
    x_denoised = reverse_step(x_t, predicted_noise, t)

    # Re-noise the original audio to timestep t-1
    original_noised = add_noise(original_audio, torch.randn_like(original_audio), t - 1)

    # Combine: keep original in unmasked regions, use generated in masked regions
    x_next = mask * x_denoised + (1 - mask) * original_noised
    return x_next

# Example: fill a 2-second gap starting at 5 seconds
sample_rate = 44100
mask = torch.zeros(1, 1, total_samples)
gap_start = 5 * sample_rate
gap_end = 7 * sample_rate
mask[:, :, gap_start:gap_end] = 1.0  # generate this region
```

---

## Cross-Modal Editing

AI enables editing music across modalities — changing properties that traditionally require re-recording:

### Style Transfer

Change the genre or instrumentation while preserving melody and structure:

```python
# Workflow: separate → re-generate → recombine
# 1. Separate the original into stems
stems = separate(original_mix)  # drums, bass, other, vocals

# 2. Re-generate specific stems with new style
new_drums = generate_with_melody(
    prompt="jazz brushed drums with swing rhythm",
    melody_reference=stems["drums"]
)

# 3. Recombine
new_mix = stems["vocals"] + new_drums + stems["bass"] + stems["other"]
```

### Tempo Modification

```python
import librosa

# Time-stretch without pitch change
y, sr = librosa.load("stem.wav")
y_fast = librosa.effects.time_stretch(y, rate=1.2)   # 20% faster
y_slow = librosa.effects.time_stretch(y, rate=0.8)   # 20% slower

# Pitch shift without tempo change
y_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)    # up 2 semitones
y_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)  # down 3 semitones
```

---

## Key Concepts

- **Source Separation**: Decomposing a mixed audio signal into individual instrument tracks (stems). Modern deep learning models achieve near-studio quality for 4-stem separation.
- **Hybrid Architecture**: Demucs processes audio in both the time domain (waveform) and frequency domain (spectrogram) simultaneously, combining strengths of both representations.
- **Music Inpainting**: Using generative models to fill in missing or masked regions of audio while maintaining coherence with surrounding context.
- **Style Transfer**: Changing the musical style, genre, or instrumentation of audio while preserving structural elements like melody and rhythm.
- **Stems**: Individual instrument tracks separated from a mix (typically: vocals, drums, bass, other/accompaniment).

---

## Further Reading

- Rouard et al., "Hybrid Transformers for Music Source Separation" (Demucs, Meta, 2023)
- Hennequin et al., "Spleeter: A Fast and Efficient Music Source Separation Tool" (Deezer, 2020)
- Lugmayr et al., "RePaint: Inpainting Using Denoising Diffusion Probabilistic Models" (2022)
- Défossez et al., "Music Source Separation in the Waveform Domain" (2019)
