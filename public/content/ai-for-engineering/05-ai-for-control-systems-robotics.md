---
title: "AI for Control Systems and Robotics"
difficulty: intermediate
topic: ai-for-engineering
order: 5
estimatedTime: "30 minutes"
summary: "Examines reinforcement learning for control systems, model predictive control with learned models, and sim-to-real transfer for robotics."
---

# AI for Control Systems and Robotics

## Overview

Control systems govern how machines maintain desired states — keeping a drone stable, guiding a robot arm to a target, or maintaining temperature in a chemical reactor. Classical control relies on mathematical models (transfer functions, state-space equations) that must be derived from first principles. **Reinforcement learning and learned models are expanding the frontier of what control systems can do**, enabling robots to adapt to unstructured environments, learn from experience, and transfer knowledge from simulation to reality.

This lesson covers RL for control, model predictive control with learned models, and sim-to-real transfer.

---

## Reinforcement Learning for Control

Reinforcement learning frames control as an agent learning to maximize cumulative reward through interaction. The agent observes the system state, takes actions, receives rewards, and updates its policy $\pi_\theta(a|s)$.

### The Control Problem as MDP

A control system is a Markov Decision Process:

- **State** $s \in \mathcal{S}$: positions, velocities, sensor readings
- **Action** $a \in \mathcal{A}$: torques, voltages, setpoints
- **Transition** $P(s'|s,a)$: system dynamics (often unknown)
- **Reward** $r(s,a,s')$: performance signal

The goal is to find a policy $\pi^*$ that maximizes expected return $J(\pi) = \mathbb{E}_\pi[\sum_{t=0}^T \gamma^t r_t]$.

### Deep RL Algorithms

Two families dominate robotics control:

**Policy Gradient (TRPO, PPO)**: Directly optimizes the policy by estimating gradients:

$$J(\pi_\theta) \approx \mathbb{E}_t\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t\right]$$

```python
import torch
import torch.nn as nn
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim * 2)  # mean and log_std per action
        )

    def forward(self, state):
        output = self.actor(state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
```

**Actor-Critic (SAC, TD3)**: Combines policy gradient with a learned value function for lower variance:

```python
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.target_critic1 = CriticNetwork(state_dim, action_dim)
        self.target_critic2 = CriticNetwork(state_dim, action_dim)
        self.alpha = torch.tensor(0.2, requires_grad=True)

    def update(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Critic loss
        with torch.no_grad():
            next_actions, _ = self.actor.get_action(next_states)
            min_next_q = torch.min(
                self.target_critic1(next_states, next_actions),
                self.target_critic2(next_states, next_actions)
            )
            target_q = rewards + (1 - dones) * 0.99 * min_next_q

        q1_loss = nn.MSELoss()(self.critic1(states, actions), target_q)
        q2_loss = nn.MSELoss()(self.critic2(states, actions), target_q)

        # Actor loss
        new_actions, log_pi = self.actor.get_action(states)
        q_values = torch.min(
            self.critic1(states, new_actions),
            self.critic2(states, new_actions)
        )
        actor_loss = (self.alpha * log_pi - q_values).mean()

        # Update networks...
```

---

## Model Predictive Control with Learned Models

Model Predictive Control (MPC) solves an optimization problem at each timestep:

$$\min_{\mathbf{u}_{0:T-1}} \sum_{t=0}^{T-1} \ell(\mathbf{x}_t, \mathbf{u}_t) \quad \text{s.t.} \quad \mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$$

Classical MPC requires an accurate model $f$. When the true system is unknown or too complex, **learned dynamics models** fill the gap:

```python
class LearnedDynamicsModel(nn.Module):
    """Neural network dynamics: x_{t+1} = f(x_t, u_t)."""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, state, action):
        xa = torch.cat([state, action], dim=-1)
        delta = self.net(xa)
        return state + delta  # Residual prediction
```

Training uses supervised learning on transition data $(\mathbf{x}_t, \mathbf{u}_t, \mathbf{x}_{t+1})$:

```python
def train_dynamics(model, dataset, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for (state, action, next_state) in dataset:
            pred_next = model(state, action)
            loss = nn.MSELoss()(pred_next, next_state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

Once trained, the learned model replaces the true dynamics in the MPC optimization — enabling **data-driven MPC** that works when first-principles models are unavailable.

---

## Sim-to-Real Transfer

A fundamental challenge: RL policies trained in simulation rarely transfer to the real world. The **sim-to-real gap** arises from model mismatch (friction, latency, sensor noise).

### Domain Randomization

Domain randomization (Tobin et al., 2017) trains across a distribution of simulated environments:

```python
def randomize_env(env, domain_rand_params):
    """Randomize physics parameters during training."""
    env.gravity = domain_rand_params['gravity'] * np.random.uniform(0.9, 1.1)
    env.friction = domain_rand_params['friction'] * np.random.uniform(0.8, 1.2)
    env.action_delay = int(domain_rand_params['action_delay'])
    env.sensor_noise = domain_rand_params['sensor_noise'] * np.random.uniform(0.5, 2.0)
```

The policy learns to be robust across this distribution, implicitly compensating for real-world mismatch.

### System Identification + Fine-Tuning

A complementary approach: identify the real-world dynamics from data and fine-tune the simulation:

```python
def system_identification(real_robot_data, sim_model):
    """Align simulator to real robot."""
    for _ in range(num_iterations):
        # Collect rollouts in real robot (limited samples!)
        real_transitions = collect_real_robot_data(n_samples=1000)

        # Update simulator parameters to match real data
        loss = mse(sim_model(real_transitions.state, real_transitions.action),
                   real_transitions.next_state)
        sim_model.optimizer.zero_grad()
        loss.backward()
        sim_model.optimizer.step()
```

### RL for Locomotion: Examples

| System | Approach | Result |
|--------|----------|--------|
| Boston Dynamics Atlas | Model-based RL + trajectory optimization | Dynamic locomotion and parkour |
| Stanford Doggo/Mini Cheetah | Sim-to-real with system ID | Agile quadruped locomotion |
| OpenAI Shadow Hand | RL + domain randomization | In-hand object manipulation |

---

## Key Takeaways

- Reinforcement learning frames control as learning from experience, enabling robots to adapt without explicit models.
- Policy gradient (PPO) and actor-critic (SAC, TD3) algorithms dominate robotic RL.
- Learned dynamics models enable data-driven MPC when first-principles models are unavailable.
- Sim-to-real gap is the central challenge — domain randomization and system identification are the primary solutions.

---

## Further Reading

- Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv 2017)
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018)
- Tobin et al., "Domain Randomization for Transferring Deep Neural Networks" (IROS 2017)
- Hwangbo et al., "Learning agile and dynamic motor skills for legged robots" (Science Robotics 2019)
- Kalashnikov et al., "Qt-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation" (2018)
