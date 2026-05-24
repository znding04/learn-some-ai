---
title: "Building Your First AI Music Generator"
level: beginner
topic: ai-for-music-and-creative-arts
order: 5
---

# Building Your First AI Music Generator

## Overview

The previous lessons covered the theory behind AI music generation — representations, diffusion models, transformers, and vocoders. Now it is time to get hands-on. In this lesson, you will generate music using pre-trained models, explore prompt engineering techniques specific to music, and build a simple pipeline that takes a text description and produces an audio file.

The barrier to generating AI music has dropped dramatically. In 2022, generating music with AI required deep expertise in machine learning, access to expensive GPU clusters, and weeks of model training. Today, pre-trained models like MusicGen can run on a consumer laptop GPU, and commercial APIs from Suno and Stability AI let you generate full songs with a single API call. The challenge has shifted from "can we generate music?" to "how do we generate the music we actually want?"

This lesson focuses on three practical skills: (1) using pre-trained open-source models locally, (2) crafting effective music prompts, and (3) building a simple generation pipeline with control over musical attributes like genre, mood, tempo, and instrumentation.

---

## Using Pre-Trained Models: MusicGen

Meta's MusicGen is the most accessible open-source music generation model. It comes in three sizes (300M, 1.5B, 3.3B parameters) and can run on a single GPU with 8GB+ VRAM.

### Setup

```bash
# Install audiocraft (Meta's music generation library)
pip install audiocraft

# For audio playback in notebooks
pip install IPython
```

### Basic Generation

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

# Load a pre-trained model
# Options: "facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-large"
model = MusicGen.get_pretrained("facebook/musicgen-small")

# Configure generation parameters
model.set_generation_params(
    duration=10,         # output duration in seconds
    top_k=250,           # restrict sampling to top 250 tokens
    temperature=1.0,     # 1.0 = default, higher = more random
    cfg_coef=3.0,        # classifier-free guidance coefficient
)

# Generate from a text description
descriptions = ["upbeat electronic dance music with synthesizer leads and a four-on-the-floor kick drum"]
wav = model.generate(descriptions)

# Save the result
audio_write("my_first_generation", wav[0].cpu(), model.sample_rate, strategy="loudness")
print(f"Saved audio: {wav.shape[-1] / model.sample_rate:.1f} seconds at {model.sample_rate} Hz")
```

### Batch Generation with Multiple Prompts

```python
# Generate multiple clips in a single batch
descriptions = [
    "gentle acoustic guitar fingerpicking in the style of folk music",
    "aggressive heavy metal riff with distorted electric guitar and double bass drums",
    "lo-fi hip hop beat with vinyl crackle and mellow piano chords",
    "orchestral film score with dramatic brass and sweeping strings",
]

wavs = model.generate(descriptions)

for i, (wav, desc) in enumerate(zip(wavs, descriptions)):
    audio_write(f"batch_{i}", wav.cpu(), model.sample_rate, strategy="loudness")
    print(f"Clip {i}: {desc[:50]}...")
```

### Melody-Conditioned Generation

MusicGen can also generate music that follows a given melody:

```python
import torchaudio

# Load a reference melody (hummed, whistled, or played)
melody_waveform, sr = torchaudio.load("my_melody.wav")

# Generate music that follows this melody but with different instrumentation
descriptions = ["jazz piano trio arrangement with upright bass and brushed drums"]
wav = model.generate_with_chroma(
    descriptions,
    melody_waveform[None],  # add batch dimension
    sr,
    progress=True
)
audio_write("melody_conditioned", wav[0].cpu(), model.sample_rate, strategy="loudness")
```

---

## Prompt Engineering for Music

Just as prompt engineering is critical for text and image generation, crafting effective music prompts significantly impacts output quality. Music prompts work best when they specify multiple musical dimensions:

### Anatomy of a Good Music Prompt

```
[Genre] + [Mood/Energy] + [Instruments] + [Tempo/Rhythm] + [Production Style]

Examples:
"mellow jazz piano trio with brushed drums and walking bass, relaxed evening feel"
"high-energy EDM drop with supersaw synths, side-chain compression, 128 BPM"
"ambient drone music with reverb-heavy pads, slow evolving textures, ethereal"
"90s boom bap hip hop beat with chopped soul samples, vinyl warmth, head-nodding groove"
```

### What Works

| Dimension | Good Prompts | Poor Prompts |
|---|---|---|
| Genre | "neo-soul R&B", "progressive house" | "good music", "popular song" |
| Instruments | "Rhodes piano, fingerpicked acoustic guitar" | "some instruments" |
| Mood | "melancholic, introspective, rainy day" | "sad" |
| Tempo | "slow waltz tempo", "driving 140 BPM" | "fast" |
| Production | "lo-fi with tape saturation", "clean studio recording" | "professional" |

### Temperature and Guidance Exploration

```python
import itertools

# Explore how generation parameters affect output
temperatures = [0.7, 1.0, 1.3]
cfg_coefs = [2.0, 3.5, 5.0]

description = "cinematic piano melody with gentle strings"

for temp, cfg in itertools.product(temperatures, cfg_coefs):
    model.set_generation_params(duration=8, temperature=temp, cfg_coef=cfg)
    wav = model.generate([description])
    audio_write(f"explore_t{temp}_cfg{cfg}", wav[0].cpu(), model.sample_rate)
    print(f"temp={temp}, cfg={cfg} — generated")
```

Lower temperatures produce more predictable, "safer" music. Higher CFG coefficients make the output adhere more closely to the prompt but can reduce naturalness.

---

## Building a Simple Generation Pipeline

Here is a complete pipeline that generates music, adds fade-in/out, and exports in multiple formats:

```python
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import numpy as np

class MusicPipeline:
    def __init__(self, model_size="small"):
        self.model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}")
        self.sr = self.model.sample_rate
    
    def generate(self, prompt, duration=15, temperature=1.0, cfg=3.0):
        """Generate audio from a text prompt."""
        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            cfg_coef=cfg,
            top_k=250,
        )
        wav = self.model.generate([prompt])
        return wav[0, 0].cpu()  # (samples,)
    
    def add_fade(self, audio, fade_in=0.5, fade_out=1.0):
        """Apply fade-in and fade-out to avoid clicks."""
        n_in = int(fade_in * self.sr)
        n_out = int(fade_out * self.sr)
        
        fade_in_curve = torch.linspace(0, 1, n_in)
        fade_out_curve = torch.linspace(1, 0, n_out)
        
        audio[:n_in] *= fade_in_curve
        audio[-n_out:] *= fade_out_curve
        return audio
    
    def normalize(self, audio, target_db=-14.0):
        """Loudness normalization."""
        rms = torch.sqrt(torch.mean(audio ** 2))
        target_rms = 10 ** (target_db / 20)
        audio = audio * (target_rms / (rms + 1e-8))
        return torch.clamp(audio, -1.0, 1.0)
    
    def run(self, prompt, output_path="output", duration=15):
        """Full pipeline: generate → fade → normalize → save."""
        print(f"Generating: {prompt}")
        audio = self.generate(prompt, duration=duration)
        audio = self.add_fade(audio)
        audio = self.normalize(audio)
        
        # Save as WAV
        torchaudio.save(f"{output_path}.wav", audio.unsqueeze(0), self.sr)
        print(f"Saved: {output_path}.wav ({duration}s)")
        return audio

# Usage
pipeline = MusicPipeline("small")
audio = pipeline.run(
    "warm lo-fi hip hop beat with jazzy piano chords and vinyl crackle",
    output_path="lofi_beat",
    duration=20
)
```

---

## Key Concepts

- **Pre-trained Models**: Models already trained on large music datasets that can generate music immediately. MusicGen, Stable Audio, and commercial APIs provide this.
- **Prompt Engineering**: The art of crafting text descriptions that guide AI music generation toward the desired output. Specificity across multiple musical dimensions yields the best results.
- **Classifier-Free Guidance (CFG) Coefficient**: Controls how strongly the model follows the text prompt. Higher values = more prompt-faithful but potentially less natural.
- **Temperature**: Controls randomness in token sampling. Lower = more deterministic and repetitive; higher = more varied but potentially less coherent.
- **Melody Conditioning**: Providing a reference melody that the model follows while generating new instrumentation and arrangement.

---

## Further Reading

- Audiocraft documentation: https://github.com/facebookresearch/audiocraft
- Copet et al., "Simple and Controllable Music Generation" (2023)
- Prompt engineering guides for Suno: community-maintained at r/SunoAI
- Stability AI Stable Audio documentation
