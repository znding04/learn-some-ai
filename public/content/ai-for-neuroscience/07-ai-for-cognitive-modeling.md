---
title: "AI for Cognitive Modeling"
difficulty: intermediate
topic: ai-for-neuroscience
order: 7
estimatedTime: "15 minutes"
summary: "Explores how AI is used to build and test computational models of cognition, including attention, working memory, decision-making, and reinforcement learning in the brain."
---
# AI for Cognitive Modeling

## Overview

Cognitive modeling asks: what computational processes produce human (or animal) behavior? How do we attention, remember, decide, and learn? AI has become a central tool for building and testing cognitive theories, because it forces theories to be precise enough to simulate behavior and generate predictions.

**Classical cognitive architectures** (ACT-R, SOAR) were built by cognitive scientists before the deep learning era. They decomposed cognition into distinct modules (visual buffer, declarative memory, production rules, etc.) with hand-tuned parameters. While influential, these models were limited in their expressiveness and scalability.

**Deep learning-based cognitive models** use neural networks trained on cognitive tasks (e.g., sequential decision making, working memory tasks, language understanding) and compare the model's behavior to human behavior. If a trained network reproduces human behavioral signatures (e.g., reaction time distributions, error patterns, learning curves), it suggests the architecture captures something real about cognitive computation.

**Probabilistic cognitive models** combine Bayesian inference with cognitive representations. Humans appear to perform approximate Bayesian inference in many domains — perception, motor control, causal reasoning. AI models that implement probabilistic computation can explain why humans make the errors they do (as suboptimal inferences under uncertainty) and why performance improves with practice (as learning a better internal model).

Key cognitive phenomena being modeled with AI include:

- **Attention**: Visual attention follows specific patterns (saccades, inhibited return); transformer-based models with attention mechanisms directly parallel human selective attention
- **Working memory**: The limited-capacity buffer for maintaining information over seconds; modeled with LSTM-like recurrent mechanisms with capacity constraints
- **Reinforcement learning in the brain**: Dopamine signals encode reward prediction errors — the same signal used in RL algorithms like Q-learning and policy gradient
- **Mentalizing / Theory of Mind**: The ability to reason about others' mental states; modeled with graph neural networks over social interaction graphs
- **Categorization and concepts**: How humans form conceptual categories; connectionist models (embedding-based) vs symbolic models (prototype-based)

## Key Concepts

- **Cognitive architecture**: A complete model of the mind's computational structure (e.g., ACT-R, SPAUN)
- **Reward prediction error (RPE)**: The difference between received and expected reward; encoded by dopamine neurons; the error signal in temporal difference (TD) learning
- **Prospect theory**: A descriptive model of human decision-making under risk, featuring loss aversion and probability weighting
- **Bayesian brain hypothesis**: The idea that the brain performs approximate Bayesian inference, maintaining probabilistic representations of world states
- **Neural manifold for cognition**: The hypothesis that cognitive variables (e.g., decision evidence, memory strength) are represented as coordinates on a low-dimensional neural manifold
- **Representational similarity analysis (RSA)**: Comparing mental representations across brain areas or between brain and model activations using similarity matrices
- **Optimal transport**: A mathematical framework for comparing probability distributions; used to compare model and human behavioral distributions

## Code Examples

```python
"""
Simulating a drift diffusion model (DDM) of decision-making
DDM explains reaction time distributions in perceptual decisions.
"""
import numpy as np
import matplotlib.pyplot as plt

def drift_diffusion(T, dt=0.001, v=0.5, a=1.0, z=0.5, s=0.1, n_simulations=10000):
    """
    Simulate decisions using the drift diffusion model.

    Parameters:
    - T: maximum decision time (s)
    - dt: time step (s)
    - v: drift rate (evidence accumulation speed)
    - a: boundary separation (decision confidence threshold)
    - z: starting point bias (relative to 0)
    - s: noise standard deviation (within-trial variability)

    Returns: reaction times and choices for each simulation
    """
    n_steps = int(T / dt)
    rts = []
    choices = []

    for _ in range(n_simulations):
        evidence = np.zeros(n_steps)
        t = 0
        # Random starting point within the boundary
        evidence[0] = np.random.uniform(-z*a, (1-z)*a)

        for t in range(n_steps):
            # Wiener process: drift + Gaussian noise
            if t > 0:
                evidence[t] = evidence[t-1] + v*dt + s*np.sqrt(dt)*np.random.randn()

            # Check decision boundary
            if evidence[t] >= a:
                rts.append(t * dt)
                choices.append(1)  # upper boundary = "yes"
                break
            elif evidence[t] <= 0:
                rts.append(t * dt)
                choices.append(0)  # lower boundary = "no"
                break
        else:
            rts.append(T)
            choices.append(np.random.choice([0, 1]))

    return np.array(rts), np.array(choices)

# Simulate with moderate evidence accumulation
rts, choices = drift_diffusion(T=2.0, v=0.8, a=1.0)

print(f"Mean RT: {rts.mean()*1000:.0f} ms")
print(f"Mean accuracy: {choices.mean()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(rts[choices==1], bins=50, alpha=0.7, label='Choice A')
axes[0].hist(rts[choices==0], bins=50, alpha=0.7, label='Choice B')
axes[0].set_xlabel('Reaction time (s)')
axes[0].set_ylabel('Count')
axes[0].set_title('RT distribution by choice')
axes[1].scatter(range(len(rts)), rts, alpha=0.1, s=1)
axes[1].set_xlabel('Trial')
axes[1].set_ylabel('RT (s)')
axes[1].set_title('RT across trials')
plt.tight_layout()
plt.savefig('/tmp/ddm.png', dpi=100)
plt.close()
print("Saved DDM figure")
```

The DDM is a bridge between cognitive theory and neural data: the drift rate $v$ correlates with neural evidence accumulation in parietal cortex, and the boundary $a$ relates to response caution mediated by prefrontal cortex.

## Further Reading

- [ACT-R cognitive architecture](https://act-r.psy.cmu.edu/)
- [PsyNet: Bayesian cognitive modeling in PyTorch](https://github.com/sjblom/pysnetwork)
- [DDM fitting with HDDM](http://ski.clps.brown.edu/hddm_docs/)