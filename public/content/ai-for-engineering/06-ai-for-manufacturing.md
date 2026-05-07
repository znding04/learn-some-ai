---
title: "AI for Manufacturing and Additive Manufacturing"
level: intermediate
topic: ai-for-engineering
order: 6
---

# AI for Manufacturing and Additive Manufacturing

## Overview

Manufacturing is the backbone of modern economies — producing everything from microchips to automobiles to medical devices. The industry faces pressure to reduce costs, improve quality, minimize waste, and respond quickly to changing demand. AI is being deployed across the manufacturing value chain: **predictive maintenance** (anticipating machine failures), **quality control** (detecting defects at line speed), **process optimization** (tuning parameters in real time), and **additive manufacturing** (3D printing with AI-guided process control).

This lesson covers these four application areas with emphasis on practical ML approaches.

---

## Predictive Maintenance

Unplanned downtime is one of the largest costs in manufacturing. A metal cutting machine going down mid-batch can cost tens of thousands of dollars in scrap, rework, and lost throughput. **Predictive maintenance** aims to forecast failures before they occur, enabling scheduled intervention.

### Remaining Useful Life Estimation

Remaining Useful Life (RUL) estimation predicts how many operational hours remain before a machine component fails. Two main approaches:

**1. Survival analysis**: Models time-to-failure using the hazard function:

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t | T \geq t)}{\Delta t}$$

Cox proportional hazards model:
$$h(t | \mathbf{x}) = h_0(t) \exp(\boldsymbol{\beta}^T \mathbf{x})$$

where $h_0(t)$ is the baseline hazard and $\mathbf{x}$ is the feature vector (vibration amplitude, temperature, usage hours).

**2. Deep learning for RUL**: LSTM networks trained on sensor time series:

```python
import torch
import torch.nn as nn

class RULPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Positive RUL output
        )
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # Final hidden state
        rul = self.fc(last_hidden)
        return rul.squeeze(-1)
```

### Anomaly Detection

When labeled failure data is scarce (failures are rare!), anomaly detection methods are preferred:

- **Autoencoders**: Train on normal operating data; reconstructino error spikes signal anomalies.
- **Isolation Forests**: Tree-based method that isolates anomalies efficiently.
- **One-Class SVM**: Learns the boundary of normal operation.

---

## Computer Vision for Quality Control

Manufacturing lines run at high speeds — hundreds of parts per minute on a semiconductor assembly line. Human inspectors cannot keep pace, and rule-based vision systems miss subtle defects. **Deep learning for visual inspection** achieves superhuman accuracy.

### Defect Classification with CNNs

```python
import torch
import torch.nn as nn

class DefectClassifier(nn.Module):
    def __init__(self, num_defect_classes=10):
        super().__init__()
        # Use pretrained ResNet backbone
        self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.backbone.fc = nn.Linear(512, num_defect_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
    def train_with_angles(self, train_loader, val_loader, epochs=50):
        """Use data augmentation (rotation, flip) for defect orientation invariance."""
        for epoch in range(epochs):
            for images, labels in train_loader:
                # Random rotation by 90° increments
                images = torch.rot90(images, torch.randint(0, 4, (images.shape[0],)), dims=[2,3])
                outputs = self(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()
```

### Semantic Segmentation for Surface Defects

For defects that span regions (scratches, dents, cracks), semantic segmentation pinpoints exact locations:

```python
class UNetDefectSegmentation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        
        # Decoder
        self.dec2 = self._block(256 + 128, 128)
        self.dec1 = self._block(128 + 64, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d2 = self.dec2(torch.cat([self.upsample(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))
```

---

## Additive Manufacturing Process Optimization

Additive manufacturing (3D printing) builds parts layer by layer. Unlike subtractive manufacturing (CNC machining), AM creates objects with internal geometries impossible by traditional methods — but the process is highly sensitive to parameters: laser power, scan speed, hatch spacing, powder properties.

### In-Situ Defect Detection

During printing, melt pool monitoring (infrared cameras) provides real-time signals. ML models analyze melt pool signatures to detect porosity and lack-of-fusion defects before they propagate:

```python
class MeltPoolClassifier(nn.Module):
    def __init__(self, input_channels=1, seq_len=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64 * 64, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)  # Normal vs Defect
    
    def forward(self, x):
        # x: [batch, seq_len, H, W]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(2).transpose(1, 2)  # [batch, seq, features]
        lstm_out, _ = self.lstm(x)
        return torch.softmax(self.fc(lstm_out[:, -1]), dim=1)
```

### Parameter Optimization with Bayesian Optimization

Finding the optimal AM parameter set (maximizing density, minimizing surface roughness) requires expensive experiments:

```python
from skopt import gp_minimize
from skopt.space import Real

def optimize_am_params():
    dimensions = [
        Real(100, 400, name='laser_power_W'),
        Real(500, 3000, name='scan_speed_mm_s'),
        Real(50, 150, name='hatch_spacing_um'),
        Real(20, 80, name='layer_thickness_um'),
    ]
    
    def objective(params):
        density, roughness = run_am_experiment(
            laser_power=params[0],
            scan_speed=params[1],
            hatch_spacing=params[2],
            layer_thickness=params[3]
        )
        return -(0.6 * density + 0.4 * (1 / (roughness + 1e-6)))
    
    result = gp_minimize(objective, dimensions, n_calls=50, random_state=42)
    return result.x
```

---

## Digital Twins for Manufacturing

A **digital twin** is a continuously updated virtual replica of a physical asset, driven by real-time sensor data. In manufacturing, digital twins enable:

1. **What-if simulation**: Test process changes before applying them to the real line
2. ** Predictive quality**: Correlate process parameters with final part quality
3. **Maintenance planning**: Integrate RUL predictions with production scheduling

```mermaid
flowchart LR
    A["Physical Machine"] -->|"Sensor data"| B["Digital Twin"]
    B -->|"Simulation"| C["Predictive Insights"]
    C -->|"Decision| D["Production Planning"]
    D -->|"Process changes| A
```

---

## Key Takeaways

- Predictive maintenance uses survival analysis and LSTM networks to forecast machine failures from sensor data, reducing unplanned downtime.
- Computer vision (CNNs, U-Net) for quality control achieves line-speed defect detection with accuracy exceeding human inspectors.
- Additive manufacturing benefits from in-situ melt pool monitoring and Bayesian optimization of process parameters.
- Digital twins provide continuous virtual replicas of manufacturing assets, enabling what-if simulation and predictive quality.

---

## Further Reading

- Lei et al., "Deep Learning for Predictive Maintenance" (IEEE Access 2020)
- Wang et al., "机会学习 for Remaining Useful Life Prediction" (Mechanical Systems and Signal Processing)
- Ren et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
- ISO/ASTM Standard for Additive Manufacturing (ISO/ASTM 52900)
- Tao et al., "Digital Twin-Driven Product Design" (CIRP Annals 2019)
