---
title: "Neural Decoding and Brain-Computer Interfaces"
difficulty: intermediate
topic: ai-for-neuroscience
order: 3
estimatedTime: "15 minutes"
summary: "Covers neural decoding methods that reconstruct sensory experiences and motor intentions from brain activity, and their application in brain-computer interfaces."
---
# Neural Decoding and Brain-Computer Interfaces

## Overview

Neural decoding is the process of reconstructing sensory experiences, motor intentions, or cognitive states from recorded brain activity. It is the inverse of the encoding problem — where encoding asks "how does a brain region represent X?", decoding asks "can we recover X from the brain activity pattern?"

The classic neural decoding experiment goes like this: a subject watches a video or imagines moving their hand while electrodes record neural firing. A decoder is trained to map the recorded neural activity (the input) to the known stimulus or intention (the target). Once trained, the decoder can predict what the subject is seeing or intending from neural data alone — without any behavioral output.

Brain-computer interfaces (BCIs) are the engineering application of neural decoding. A BCI reads brain signals, decodes the user's intent, and translates it into commands for a computer, prosthetic limb, or other device. The most well-studied BCIs are motor BCIs that decode movement intention from the motor cortex, enabling people with paralysis to control robotic arms or computers.

Neural signals for decoding come from multiple sources:
- **Single-unit recordings**: Microelectrodes that record action potentials from individual neurons. The signal is a spike train — a series of precise timestamps. Requires surgical implantation.
- **Local Field Potentials (LFP)**: Lower-frequency electrical signals reflecting the summed activity of local neural populations.
- **Electrocorticography (ECoG)**: Arrays of electrodes placed on the surface of the brain (under the skull), providing higher spatial resolution than scalp EEG with lower invasiveness than single units.
- **EEG**: Non-invasive but noisy; useful for BCIs based on ERPs (e.g., P300 speller) and sensorimotor rhythms (mu rhythm).
- **fMRI**: Indirect BOLD signal; used for decoding cognitive states in research settings but too slow for real-time BCIs.

Modern neural decoders range from simple linear filters (Wiener filter, Kalman filter) to deep recurrent networks. The choice depends on the signal type, the complexity of the decoded variable, and the requirement for online real-time operation.

## Key Concepts

- **Spike train**: A sequence of timestamps marking when a neuron fired an action potential. The fundamental unit of neural data.
- **Firing rate**: Number of spikes per second, computed by binning spike times into time windows (e.g., 50ms bins)
- **Receptive field**: The specific sensory stimulus feature (e.g., a particular orientation or spatial location) that causes a neuron to fire
- **Motor decoding**: Predicting movement parameters (velocity, direction, grip force) from motor cortex activity
- **Kalman filter**: A linear recursive decoder that models both the dynamics of the movement and the observation noise in the neural data
- **Shared manifold**: The hypothesis that neural population activity across many conditions lies on a low-dimensional manifold — a geometric constraint that decoders can exploit
- **Closed-loop BCI**: A BCI where the decoded output affects the feedback the subject receives, creating a bidirectional loop

## Code Examples

```python
"""
Neural decoding: Wiener filter for motor cortical activity
Simulating neural population activity during reaching movements.
"""
import numpy as np

np.random.seed(42)
n_neurons = 100
n_timepoints = 500
n_trials = 50

# Simulate: 100 neurons recorded from motor cortex during reaching
# Each neuron has a preferred direction (the direction it fires most for)
preferred_directions = np.random.uniform(0, 2*np.pi, n_neurons)

# Generate spike counts in 50ms bins for a single trial
# Firing rate ~ base_rate * exp(similarity to preferred direction)
def generate_reaching_activity(preferred_dirs, reach_direction, n_bins=100):
    base_rate = 15  # spikes per second baseline
    tuning_strength = 3.0
    rates = base_rate * np.exp(tuning_strength * np.cos(preferred_dirs - reach_direction))
    spike_counts = np.random.poisson(rates * 0.05)  # 50ms bins
    return spike_counts

# Generate trials with different reach directions
directions = np.random.uniform(0, 2*np.pi, n_trials)
all_spike_counts = np.array([generate_reaching_activity(preferred_directions, d) for d in directions])

# Wiener filter decoding: find linear weights that map spike counts to direction
# Simple version: use pseudoinverse to solve W * spikes = direction
X = all_spike_counts  # shape: (n_trials, n_neurons)
y = directions       # shape: (n_trials,)

# Add bias term
X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])

# Solve for weights using least squares
W = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
y_pred = X_with_bias @ W

decode_error = np.mean(np.abs(y_pred - y))
print(f"Mean absolute decoding error: {np.degrees(decode_error):.2f} degrees")
print(f"(This is a simplified single-timepoint decode — real decoders use time series)")
```

This illustrates the basic principle: neurons have tuning curves (preferred directions), and a linear decoder can recover the reach direction from population activity. More sophisticated decoders like the Kalman filter model the temporal dynamics of movement.

## Further Reading

- [BNCI Horizon 2020 database](https://bbci.de/competition/)
- [Neural Decoding Toolbox](https://github.com/KordingLab/Neural_Decoding Toolbox)
- [Buzsaki Lab neuroscience resources](https://buzsakilab.com/)