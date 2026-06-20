---
title: "AI for Computer-Aided Design"
difficulty: intermediate
topic: ai-for-engineering
order: 7
estimatedTime: "30 minutes"
summary: "Explores AI for CAD workflows including sketch understanding, parametric model automation, design search, and generative design of mechanical parts."
---

## AI for Computer-Aided Design

## Overview

Computer-Aided Design (CAD) software — SolidWorks, CATIA, Fusion 360, Onshape — is the foundational tool of modern engineering. Engineers spend a substantial fraction of their time in CAD: sketching, extruding, applying mates, checking interference, and generating manufacturing drawings. **AI is beginning to automate and accelerate CAD workflows**, from generative sketching to parametric model inference to natural language design intent.

This lesson covers four application areas: AI for sketch understanding, parametric CAD automation, design search and retrieval, and generative design of mechanical parts.

---

## AI for Sketch Understanding

Engineering sketches are the starting point of CAD models. Converting a hand-drawn sketch into a CAD feature (extrusion, revolution, sweep) requires recognizing geometric primitives and their constraints.

### Primitive Recognition

CNNs trained on annotated sketch datasets can classify geometric primitives (line, arc, circle, spline) and detect constraints (parallel, perpendicular, tangent, concentric):

```python
import torch
import torch.nn as nn

class SketchPrimitiveClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.backbone.fc = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = torch.relu(self.backbone(x))
        return self.classifier(features)

    def detect_constraints(self, primitives, keypoints):
        """
        Detect geometric constraints between recognized primitives.
        Returns constraint graph: edges = {parallel, perpendicular, tangent, ...}
        """
        constraint_edges = []
        for i, j in combinations(range(len(primitives)), 2):
            feat = torch.cat([primitives[i], primitives[j],
                             distance(keypoints[i], keypoints[j])])
            constraint_type = self.constraint_head(feat)
            if torch.argmax(constraint_type) != 0:  # Not "no constraint"
                constraint_edges.append((i, j, constraint_type))
        return constraint_edges
```

### Sketch-to-CAD Translation

Recent work uses seq2seq models to translate sketches directly to CAD operations:

```python
class SketchToCADTranslator(nn.Module):
    """
    Translates 2D sketch keypoints to CAD operations.
    Output: sequence of operations (Line, Arc, Circle, Extrude, etc.)
    """
    def __init__(self, keypoint_dim=2, hidden=256, num_ops=15):
        super().__init__()
        self.keypoint_encoder = nn.GRU(keypoint_dim, hidden, batch_first=True)
        self.operation_decoder = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.operation_head = nn.Linear(hidden, num_ops)
        self.parameter_head = nn.Linear(hidden, 4)  # dx, dy, radius, angle

    def forward(self, keypoints):
        # keypoints: [batch, seq_len, 2] (x, y per point)
        encoded, h_n = self.keypoint_encoder(keypoints)
        h_init = torch.cat([h_n[-1], h_n[-1]], dim=-1)

        # Autoregressive generation of CAD operations
        ops = []
        hidden = h_init.unsqueeze(1)
        for t in range(max_seq_len):
            output, hidden = self.operation_decoder(hidden)
            op_logits = self.operation_head(output)
            op_params = self.parameter_head(output)
            ops.append((op_logits, op_params))
        return ops
```

---

## Parametric CAD Automation

CAD models are parametric: features are defined by dimensions and geometric relationships. Changing a single dimension triggers a cascade of recomputations. **ML can learn to predict parametric changes that achieve desired geometric outcomes**.

### Inverse CAD: From Shape to Parameters

Given a 3D point cloud or image of a mechanical part, inverse CAD infers the parametric feature tree:

```python
class InverseCADNet(nn.Module):
    """Infers CAD parametric feature tree from 3D point cloud."""
    def __init__(self, num_feature_types=20):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.AdaptiveAvgPool1d(1)
        )
        self.feature_predictor = nn.Linear(256, num_feature_types)  # Sketch plane, extrude, etc.
        self.parameter_predictor = nn.Linear(256, 50)  # Dimensional parameters
        self.construction_predictor = nn.Linear(256, 20)  # Mates, constraints

    def forward(self, point_cloud):
        # point_cloud: [batch, 3, N]
        encoded = self.point_encoder(point_cloud).squeeze(-1)
        feature_probs = torch.softmax(self.feature_predictor(encoded), dim=-1)
        parameters = self.parameter_predictor(encoded)
        construction = torch.softmax(self.construction_predictor(encoded), dim=-1)
        return feature_probs, parameters, construction
```

### Design Intent Modeling

When an engineer modifies a dimension, they often have implicit intent (e.g., "I want to increase stiffness without changing the mass"). LLMs can interpret natural language design intent and translate it to parametric changes:

```python
def interpret_design_intent(natural_language, cad_model):
    prompt = f"""
    CAD model: {cad_model.description}
    User says: "{natural_language}"

    What parametric changes achieve this intent?
    Return: {{"changed_features": [...], "new_parameters": {{...}}}}
    """
    response = llm.generate(prompt)
    return parse_cad_changes(response)
```

---

## Design Search and Retrieval

Large organizations have decades of CAD designs in PDM/PLM systems. Finding the right design to reuse or adapt is notoriously difficult. **ML-based design search** indexes CAD geometry and enables semantic retrieval.

### Shape Matching with Deep Learning

```python
class ShapeDescriptor(nn.Module):
    """
    Learns a compact descriptor for 3D CAD models.
    Similar designs have similar descriptors.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, voxel_grid):
        # voxel_grid: [batch, 1, V, V, V] (voxelized CAD model)
        features = self.encoder(voxel_grid).squeeze(-1).squeeze(-1).squeeze(-1)
        return self.projection(features)

    def retrieval_query(self, query_model, database_models, k=5):
        query_emb = self.forward(query_model)
        db_embs = torch.stack([self.forward(m) for m in database_models])
        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), db_embs)
        return torch.topk(similarities, k).indices
```

---

## Generative Modeling of Mechanical Parts

The most ambitious application: **generating new mechanical parts from specifications**. Diffusion models and autoregressive models trained on CAD datasets can produce valid 3D geometries that satisfy functional requirements.

### Point Cloud Generation

```python
class PartDiffusion(nn.Module):
    """Diffusion model for 3D point clouds of mechanical parts."""
    def __init__(self, num_points=2048, hidden=512, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.point_net = nn.Sequential(
            nn.Linear(3 + time_dim, hidden),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, x, t):
        # x: [batch, N, 3], t: [batch, 1] (diffusion timestep)
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        return self.point_net(h)

    @torch.no_grad()
    def sample(self, n_points, num_steps=100):
        """Generate point cloud via DDPM."""
        x_t = torch.randn(n_points, 3)
        for t in reversed(range(num_steps)):
            t_tensor = torch.tensor([t / num_steps]).expand(x_t.shape[0], 1)
            noise_pred = self.forward(x_t, t_tensor)
            alpha_t = self.alpha_bar[t]
            x_t = (x_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        return x_t
```

---

## Key Takeaways

- Sketch understanding uses CNNs for primitive recognition and constraint detection, enabling sketch-to-CAD translation.
- Inverse CAD infers parametric feature trees from 3D point clouds, automating the conversion of scanned parts to editable models.
- LLMs can interpret natural language design intent and translate it to parametric model changes.
- Design search with learned shape descriptors enables semantic retrieval across CAD databases.
- Diffusion models trained on CAD point clouds can generate novel mechanical parts satisfying functional specifications.

---

## Further Reading

- Li et al., "MgrNet: A Deep Learning Framework for Mechanical Drawing Recognition" (CAD 2021)
- Xu et al., "DeepCAD: A Deep Learning Framework for CAD Model Retrieval" (CVPR 2020)
- Zhou et al., "LatentCAD: Unified Representation for Engineering Design" (arXiv)
- Del形式的 et al., "SketchGraphs: A Large-Scale Dataset for Modeling Shape Constraints" (ICML 2020)
- Nichizawa et al., "3D ShapeNets: A Deep Representation for Volumetric Shapes" (CVPR 2015)
