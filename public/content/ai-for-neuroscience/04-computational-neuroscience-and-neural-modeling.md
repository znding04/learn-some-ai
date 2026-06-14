---
title: "Computational Neuroscience and Neural Modeling"
difficulty: intermediate
topic: ai-for-neuroscience
order: 4
estimatedTime: "15 minutes"
summary: "Introduces mathematical models of neurons and neural circuits, from the Hodgkin-Huxley model to mean field models, and their deep connection to AI architectures."
---

# Computational Neuroscience and Neural Modeling

## Overview

Computational neuroscience builds mathematical models of neurons and neural circuits to understand how the brain processes information. These models range from single-compartment biophysical models of individual neurons to large-scale network models of cortical columns. AI and computational neuroscience have a deeply mutual relationship: AI borrows neural-inspired architectures (CNNs from visual cortex, transformers from attention-like mechanisms), while computational neuroscience borrows AI optimization methods to fit models to data.

The foundational model of neuronal computation is the **Hodgkin-Huxley (HH) model** (Hodgkin & Huxley, 1952 — Nobel Prize). It describes how action potentials arise from the dynamics of ion channels (sodium and potassium) embedded in the neuronal membrane. The model is a system of four coupled nonlinear differential equations:

$$C_m \frac{dV}{dt} = -\bar{g}_K n^4 (V - E_K) - \bar{g}_{Na} m^3 h (V - E_{Na}) - \bar{g}_L (V - E_L) + I$$

where $V$ is membrane potential, $m, h, n$ are ion channel gating variables, $C_m$ is membrane capacitance, and $I$ is input current. The nonlinear dynamics of these equations produce action potentials — all-or-nothing electrical pulses that neurons use to communicate.

At the circuit level, researchers use **mean field models** to describe the average activity of large neural populations. These reduce thousands of neurons to a few differential equations describing population firing rates — much like how fluid dynamics reduces molecular collisions to continuous equations. Mean field models are tractable for analytical study and capture emergent phenomena like oscillations, bistability, and winner-take-all competition.

**Neural network models** draw a closer connection to AI. Artificial neural networks were originally inspired by biological neurons. Recurrent neural networks (RNNs) with firing-rate dynamics map naturally onto biological circuit models. The neural tangent kernel (NTK) theory formalizes how neural networks behave as kernel methods near initialization — bridging deep learning theory and neuroscience.

The parameter fitting problem is central: given recorded neural activity, what model parameters produce that activity? This is an inverse problem solved with gradient-based optimization, Bayesian inference, or evolutionary strategies. Modern approaches use automatic differentiation (via PyTorch/JAX) to fit large biophysical models to imaging data.

## Key Concepts

- **Membrane potential ($V_m$)**: The electrical potential difference across the neuronal membrane, typically -70mV at rest
- **Action potential**: A brief (1ms) positive spike in membrane potential that propagates down the axon; the neural code's fundamental unit
- **Ion channel**: Protein pores in the membrane that allow specific ions (Na⁺, K⁺, Ca²⁺) to flow, generating current
- **Leaky integrate-and-fire (LIF)**: A simplified neuron model that integrates input current and fires when a threshold is reached: $\tau \frac{dV}{dt} = -V + I$
- **Firing rate model**: A model where the state variable is the average firing rate rather than individual spikes
- **Synaptic weights**: The strength of connection between two neurons; modified by synaptic plasticity rules (Hebbian, STDP)
- **Neural mass model**: A model describing the mean activity of a neural population, using equations for excitatory and inhibitory pools
- **Neural tangent kernel (NTK)**: A kernel that describes how neural network predictions change under gradient descent near initialization

## Code Examples

```python
"""
Leaky Integrate-and-Fire (LIF) neuron simulation
The simplest biophysically plausible neuron model.
"""
import numpy as np
import matplotlib.pyplot as plt

def lif_neuron(I_ext, dt=0.1, t_max=200,
               V_rest=-70, V_thresh=-55, V_reset=-75,
               tau=10, R=10):
    """
    Simulate a LIF neuron receiving external current.

    Parameters:
    - I_ext: external input current (nA) over time
    - dt: time step (ms)
    - V_rest: resting potential (mV)
    - V_thresh: threshold potential (mV)
    - V_reset: reset potential after spike (mV)
    - tau: membrane time constant (ms)
    - R: membrane resistance (MOhm)
    """
    n_steps = int(t_max / dt)
    V = np.full(n_steps, V_rest)
    spikes = []

    for t in range(n_steps):
        # LIF dynamics
        dV = (-(V[t] - V_rest) + R * I_ext[t]) / tau
        V[t+1] = V[t] + dt * dV

        # Spike condition
        if V[t+1] >= V_thresh:
            spikes.append(t * dt)
            V[t+1] = V_reset

    return V[:-1], np.array(spikes)

# Simulate with a step current injection
t_max = 200  # ms
dt = 0.1
time = np.arange(0, t_max, dt)
I_ext = np.where((time > 50) & (time < 150), 2.5, 0.0)  # 2.5 nA from 50-150ms

V, spikes = lif_neuron(I_ext, dt=dt, t_max=t_max)

print(f"Number of spikes: {len(spikes)}")
print(f"Spike times (ms): {spikes}")
# The LIF neuron fires at a regular rhythm determined by its inputs
```

The LIF model is computationally efficient and captures key neuron properties: integration of synaptic input, threshold-based spiking, and refractory period. Real neurons have additional complexity (ion channel dynamics, dendritic compartmentalization, synaptic short-term plasticity), but LIF is the workhorse for large-scale network models.

## Further Reading

- [Brian2 spiking network simulator](https://brian2.readthedocs.io/)
- [NEURON simulation environment](https://neuron.yale.edu/)
- [PyNN simulator API](https://neuralensemble.org/pynn/)