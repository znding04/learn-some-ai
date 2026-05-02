# Newton's Laws and Forces

## Introduction

Forces are the agents of change in the physical world. While kinematics describes *how* objects move, **dynamics** explains *why* they move — through the action of forces. Newton's three laws form the foundation of classical mechanics.

## Newton's Three Laws

### First Law (Inertia)
An object remains at rest or in uniform motion unless acted upon by a net external force.

$$\sum \vec{F} = 0 \implies \vec{v} = \text{constant}$$

This is the principle of **inertia** — objects resist changes in their state of motion.

### Second Law (F = ma)
The acceleration of an object is directly proportional to the net force and inversely proportional to its mass.

$$\vec{F}_{\text{net}} = m\vec{a}$$

or equivalently, in terms of momentum: $\vec{F}_{\text{net}} = \frac{d\vec{p}}{dt}$

### Third Law (Action-Reaction)
For every action force, there is an equal and opposite reaction force.

$$\vec{F}_{12} = -\vec{F}_{21}$$

## Common Forces in Mechanics

### Gravity
The force exerted by Earth on objects near its surface:

$$F_g = mg$$

where $g \approx 9.8 \text{ m/s}^2$ downward.

### Normal Force
The contact force exerted by a surface, perpendicular to the surface. For a flat surface:

$$N = mg$$

(on flat ground, normal force equals weight).

### Friction
The force that opposes relative motion between surfaces.

| Type | Formula | Direction |
|------|---------|-----------|
| Static friction | $f_s \leq \mu_s N$ | Opposes attempted motion |
| Kinetic friction | $f_k = \mu_k N$ | Opposes actual motion |

**Note:** Static friction adjusts to match the applied force up to its maximum value $\mu_s N$.

## Free Body Diagrams

The essential tool for solving force problems:

1. Identify all forces acting on the object
2. Draw each force as an arrow from the object's center
3. Label the forces (F_g, N, f_k, T, etc.)
4. Apply Newton's second law in each direction

## Practice Problems

1. **A 5 kg object rests on a horizontal table with $\mu_s = 0.4$. What is the maximum static friction force before it starts moving?**
   <details><summary>Answer</summary>$f_{s,\max} = \mu_s N = \mu_s mg = 0.4 \times 5 \times 9.8 = 19.6$ N</details>

2. **A 3 kg block is pulled across a rough surface with kinetic friction $\mu_k = 0.3$ by a horizontal force of 15 N. What is the acceleration?**
   <details><summary>Answer</summary>$f_k = \mu_k N = 0.3 \times 3 \times 9.8 = 8.82$ N. Then $F_{\text{net}} = 15 - 8.82 = 6.18$ N, so $a = F_{\text{net}}/m = 6.18/3 = 2.06$ m/s²</details>

3. **A 70 kg person stands on a scale in an elevator accelerating upward at 2 m/s². What does the scale read?**
   <details><summary>Answer</summary>$N - mg = ma$, so $N = m(g+a) = 70(9.8 + 2) = 70 \times 11.8 = 826$ N (about 84 kg equivalent)</details>

## Key Takeaways

- Newton's second law $F = ma$ is the bridge between forces and motion
- Free body diagrams are essential — always draw them first
- Static friction can range from 0 to $\mu_s N$; kinetic friction is constant at $\mu_k N$
- Newton's third law pairs forces on *different* objects — never put them on the same FBD
