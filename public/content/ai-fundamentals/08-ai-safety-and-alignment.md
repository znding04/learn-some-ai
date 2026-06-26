---
title: "AI Safety and Alignment"
difficulty: intermediate
topic: ai-fundamentals
order: 8
estimatedTime: "15 minutes"
summary: "Covers AI alignment challenges including the specification problem, RLHF, Constitutional AI, mechanistic interpretability, and current safety research directions."
---
# AI Safety and Alignment

## Overview

As AI systems become more capable, ensuring they behave as intended becomes critical. AI alignment is the challenge of building AI systems whose goals and behaviors are aligned with human values and intentions. Getting this wrong could range from annoying (a chatbot that gives bad advice) to catastrophic (a powerful system pursuing misaligned objectives).

### What is AI Alignment?

Alignment means ensuring an AI system does what we actually want, not just what we literally asked for. This is harder than it sounds because:

- **Specification problem**: It's difficult to precisely define what we want. "Maximize user engagement" sounds reasonable until the AI learns that outrage maximizes clicks.
- **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure." An AI optimizing a proxy metric will find ways to game it.

Classic thought experiment: Tell an AI to "make paperclips as efficiently as possible." A sufficiently powerful, misaligned system might convert all available matter — including humans — into paperclips. The problem isn't malice; it's that the objective was underspecified.

### Core Alignment Problems

**Specification** — How do we tell the AI what we want?

Reward functions and objective specifications are inherently incomplete. Human values are complex, contextual, and sometimes contradictory. We can't write a complete specification of "be helpful and harmless."

**Robustness** — How do we ensure the AI behaves well in new situations?

A model might behave perfectly during testing but fail in unexpected ways during deployment. **Distribution shift** — when the real world differs from training data — is a major concern.

**Monitoring and Interpretability** — How do we know what the AI is doing and why?

As models grow larger, understanding their internal reasoning becomes harder. **Mechanistic interpretability** aims to reverse-engineer neural network internals to understand their computations.

**Scalable Oversight** — How do humans supervise AI systems that are faster and potentially smarter than them?

This becomes increasingly difficult as AI capabilities grow. We need methods that scale — where AI systems help humans evaluate other AI systems.

### RLHF: Reinforcement Learning from Human Feedback

RLHF is the dominant technique for aligning language models. The process:

1. **Supervised fine-tuning (SFT)**: Train the model on high-quality demonstrations of desired behavior
2. **Reward model training**: Humans rank model outputs (A > B). Train a reward model $R(x, y)$ to predict human preferences
3. **RL optimization**: Use PPO (Proximal Policy Optimization) to fine-tune the language model to maximize the reward model's score

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi}[R(x, y)] - \beta \cdot D_{KL}[\pi || \pi_{\text{ref}}]$$

The KL divergence term prevents the model from deviating too far from the original (reference) policy, avoiding reward hacking.

**Limitations of RLHF**:
- Human evaluators have inconsistent preferences
- Reward models can be gamed (reward hacking)
- Optimizing for "what sounds good to a human" ≠ being correct
- Doesn't scale to superhuman tasks (humans can't evaluate what they can't understand)

### Constitutional AI (CAI)

Developed by Anthropic, Constitutional AI reduces reliance on human labelers:

1. Start with a set of principles (a "constitution") — e.g., "be helpful, harmless, and honest"
2. Generate responses, then ask the AI to critique and revise its own responses based on the constitution
3. Use the revised responses to train a preference model
4. Fine-tune with RL against this preference model

This is more scalable than pure RLHF because the AI helps generate training signal, but the principles still come from humans.

### Current Safety Research Directions

**Red-teaming**: Systematically trying to make AI systems fail or produce harmful outputs. Essential for finding vulnerabilities before deployment.

**Mechanistic Interpretability**: Understanding what individual neurons and circuits inside neural networks are computing. Anthropic, DeepMind, and others are making progress on identifying meaningful features in large models.

**Scalable Oversight**: Techniques like debate (two AIs argue, human judges), recursive reward modeling, and constitutional approaches.

**Evaluations (Evals)**: Developing benchmarks for dangerous capabilities — can the model help with bioweapons? Can it deceive evaluators? Mandatory evals are becoming an industry norm.

**Governance**: AI labs adopting responsible scaling policies (RSPs) that define capability thresholds triggering additional safety measures.

## Key Concepts

- **Alignment**: Ensuring AI systems pursue goals that match human intentions and values
- **Specification Problem**: The difficulty of precisely defining what we want AI to do
- **RLHF**: Training AI using human preference rankings to shape behavior
- **Constitutional AI**: Using principles and AI self-critique to scale alignment
- **Mechanistic Interpretability**: Reverse-engineering neural network internals to understand reasoning
- **Red-teaming**: Adversarial testing to find failure modes before deployment

## Exercises

1. **Goodhart's Law in practice**: Give 3 examples where optimizing a proxy metric could lead to unintended behavior in AI systems.
2. **Design a constitution**: Write 5 principles you would include in a Constitutional AI system for a medical chatbot.
3. **Debate**: Should AI labs be required to share safety research? Argue both sides.
4. **Research**: Look up Anthropic's Responsible Scaling Policy. What capability levels does it define?

## Further Reading

- Amodei, D. et al. (2016). "Concrete Problems in AI Safety"
- Bai, Y. et al. (2022). "Constitutional AI: Harmlessness from AI Feedback"
- Christiano, P. et al. (2017). "Deep Reinforcement Learning from Human Preferences"
- Neel Nanda's "A Comprehensive Mechanistic Interpretability Explainer" (blog)
