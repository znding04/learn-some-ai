---
title: "AI for Electrical and Computer Engineering"
level: beginner
topic: ai-for-engineering
order: 4
---

# AI for Electrical and Computer Engineering

## Overview

Electrical and computer engineering encompasses everything from nanoscale transistors to continental power grids. Modern chips contain billions of transistors; power grids coordinate thousands of generators. The design and operation of these systems generates enormous complexity that is increasingly addressed with AI. **Chip design, circuit optimization, fault detection, and power system operation** are all being transformed by machine learning.

This lesson covers three major areas: AI for electronic design automation (EDA), AI for circuit optimization, and AI for power grid operation.

---

## AI for Electronic Design Automation

Electronic Design Automation (EDA) is the software stack that transforms a logical circuit specification into a physical chip layout ready for manufacturing. It includes synthesis (translating RTL to gate-level netlists), placement (deciding where each gate goes), routing (connecting gates with metal wires), timing analysis (ensuring signals arrive at the right time), and verification (checking correctness).

Google's Nature 2021 paper on **reinforcement learning for chip floorplanning** demonstrated that an RL agent could place circuits more efficiently than human experts. The agent was trained to minimize wirelength and congestion while satisfying timing constraints. The key innovations:

1. **Graph representation**: The chip netlist is represented as a graph where nodes are macros (memory, IP blocks) and standard cells, and edges are connections.
2. **Policy network**: A graph neural network encodes the netlist and predicts placements for each macro.
3. **Reward**: Negative weighted wirelength plus proxy costs for congestion and timing violations.

```python
import torch
import torch.nn as nn

class ChipFloorplanGNN(nn.Module):
    """GNN policy for chip floorplanning."""
    def __init__(self, node_dim=64, edge_dim=32, hidden=128):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden)
        self.edge_encoder = nn.Linear(edge_dim, hidden)
        self.message_passing = nn.ModuleList([
            nn.GATConv(hidden, hidden, heads=4) for _ in range(6)
        ])
        self.policy_head = nn.Linear(hidden, 4)  # dx, dy, flip, rotation
    
    def forward(self, node_features, edge_index, edge_features):
        # node_features: [N, node_dim]
        # edge_index: [2, E]
        # edge_features: [E, edge_dim]
        x = torch.relu(self.node_encoder(node_features))
        
        for layer in self.message_passing:
            x = layer(x, edge_index)
            x = torch.relu(x)
        
        action_logits = self.policy_head(x)
        return action_logits
```

Beyond floorplanning, AI is now applied across the entire EDA toolchain:

| Stage | AI Application |
|-------|---------------|
| Synthesis | RL for RTL-to-gate optimization |
| Placement | Graph RL for macro placement |
| Routing | GNN-based congestion prediction |
| Timing | Gradient boosting for STA signoff |
| Verification | CNN for bug detection in waveforms |

---

## Circuit Optimization with Machine Learning

### Analog Circuit Sizing

Analog circuits (amplifiers, data converters, PLLs) require hand-tuning of hundreds of device parameters — widths, lengths, and bias currents. This is a tedious, knowledge-intensive process. ML models trained on simulation data can predict circuit performance and enable optimization:

```python
import torch
import torch.nn as nn

class AnalogCircuitPredictor(nn.Module):
    """Predicts analog circuit metrics from device sizes."""
    def __init__(self, n_devices=20, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_devices, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5)  # DC gain, GBW, phase margin, noise, power
        )
    
    def forward(self, device_sizes):
        # device_sizes: [batch, n_devices] (W/L values)
        return self.net(device_sizes)
```

### Bayesian Optimization for Circuit Tuning

Given an expensive circuit simulator (SPICE), Bayesian optimization efficiently explores the design space:

```python
from skopt import gp_minimize
from skopt.space import Real, Integer

def optimize_amplifier():
    dimensions = [
        Real(1e-6, 100e-6, name='W1'),   # Input transistor width
        Real(1e-6, 10e-6, name='L1'),   # Input transistor length
        Real(1e-6, 100e-6, name='W2'),  # Load transistor width
        Real(1e-6, 10e-6, name='L2'),   # Load transistor length
    ]
    
    result = gp_minimize(
        lambda x: -simulate_amplifier(x),  # Negative because we maximize
        dimensions,
        n_calls=50,
        random_state=42,
        acq_func='EI',
        n_initial_points=10
    )
    return result.x
```

---

## AI for Power Grid Operation

Modern power grids face challenges that traditional control cannot address: renewable energy intermittency, electric vehicle charging loads, and aging infrastructure. AI is being deployed at multiple levels:

### Load Forecasting

Accurate load forecasting is essential for unit commitment and economic dispatch. Gradient boosting models (XGBoost, LightGBM) trained on historical load, weather, calendar features, and economic indicators outperform traditional time-series methods:

```python
import lightgbm as lgb
import numpy as np

def train_load_forecaster(X_train, y_train, X_test):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=500)
    
    return model.predict(X_test)
```

### Fault Detection in Transmission Lines

Transmission lines are monitored by PMUs (Phasor Measurement Units) that provide synchronized voltage and current phasors at 30-60 Hz. Graph neural networks trained on simulated fault data can classify fault types and locations:

```python
class FaultClassifierGNN(nn.Module):
    def __init__(self, n_features=12, hidden=64, n_classes=10):
        super().__init__()
        self.conv1 = GCNConv(n_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, n_classes)
    
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return torch.softmax(self.classifier(x), dim=1)
```

---

## Key Takeaways

- AI for EDA applies RL and GNNs to placement, routing, and timing optimization — Google's RL for floorplanning outperformed human experts.
- Analog circuit sizing is being automated with ML surrogate models trained on SPICE simulations, accelerated by Bayesian optimization.
- Power grid operation benefits from load forecasting (gradient boosting) and fault detection (GNNs on PMU data).
- EDA and chip design represent one of the most demanding engineering AI applications — billions of dollars of R&D depend on these tools.

---

## Further Reading

- Mirhoseini et al., "A Graph Placement Methodology for Fast Chip Design" (Nature 2021)
- Google's circuit training repository: https://github.com/google/circuit_training
- XGBoost/LightGBM for load forecasting: relevant survey papers
- Zhou et al., "Machine Learning for Electronic Design Automation" (ACM Computing Surveys 2022)
- Dalvi et al., "Bayesian Optimization for Analog Circuit Sizing" (2022)
