# Impulse and Momentum

## Introduction

Momentum is a **vector** quantity that captures the "quantity of motion" an object has. Combined with the concept of **impulse**, it provides a powerful framework for analyzing collisions and interactions between objects — often superior to working with forces directly when time intervals are known.

## Momentum

The linear momentum of an object:

$$\vec{p} = m\vec{v}$$

### Key Properties
- **Vector:** Has direction (same direction as velocity)
- ** SI unit:** kg·m/s
- For multiple particles: $\vec{p}_{\text{total}} = \sum m_i\vec{v}_i$

## Impulse

**Impulse** is the change in momentum, equal to the net force multiplied by the time interval:

$$\vec{J} = \vec{F}_{\text{net}}\Delta t = \Delta\vec{p} = \vec{p}_f - \vec{p}_i$$

This is the **Impulse-Momentum Theorem** — technically always valid, even when forces vary with time.

### Force-Time Graphs
The area under a force-time graph equals the impulse (and thus the change in momentum).

## Conservation of Momentum

For an **isolated system** (no external forces):

$$\vec{p}_{\text{total}} = \text{constant}$$

$$\sum \vec{p}_i = \sum \vec{p}_f$$

This is one of the most fundamental conservation laws in physics — it holds even when classical mechanics fails (collisions at high speeds, subatomic particles).

## Collisions

### Elastic Collision
Both momentum AND kinetic energy are conserved.

| Conserved | Formula |
|-----------|---------|
| Momentum | $m_1v_{1i} + m_2v_{2i} = m_1v_{1f} + m_2v_{2f}$ |
| Kinetic Energy | $\frac{1}{2}m_1v_{1i}^2 + \frac{1}{2}m_2v_{2i}^2 = \frac{1}{2}m_1v_{1f}^2 + \frac{1}{2}m_2v_{2f}^2$ |

### Inelastic Collision
Momentum is conserved, but kinetic energy is NOT conserved (energy converts to other forms: heat, sound, deformation).

**Perfectly inelastic** — objects stick together after collision ($v_{1f} = v_{2f} = v_f$):

$$v_f = \frac{m_1v_{1i} + m_2v_{2i}}{m_1 + m_2}$$

### Comparison

| Type | Momentum | Kinetic Energy |
|------|----------|----------------|
| Elastic | ✓ Conserved | ✓ Conserved |
| Inelastic | ✓ Conserved | ✗ Lost |
| Perfectly Inelastic | ✓ Conserved | ✗ Lost (max) |

## 2D Collisions

Momentum is a vector, so conservation applies component-wise:

$$p_{xi} = p_{xf}, \quad p_{yi} = p_{yf}$$

This allows analysis of billiard-ball collisions and other 2D scattering problems.

## Practice Problems

1. **A 1 kg ball traveling at 5 m/s collides head-on with a 2 kg ball at rest. If the collision is perfectly inelastic, what is the final velocity?**
   <details><summary>Answer</summary>$v_f = \frac{m_1v_{1i} + m_2v_{2i}}{m_1 + m_2} = \frac{1 \times 5 + 2 \times 0}{1 + 2} = \frac{5}{3} \approx 1.67$ m/s in the original direction</details>

2. **A 0.5 kg baseball is thrown at 40 m/s and hit back at 50 m/s. The bat was in contact for 0.002 s. What was the average force?**
   <details><summary>Answer</summary>$\Delta p = m(v_f - v_i) = 0.5(-50 - 40) = -45$ kg·m/s (negative because direction reversed). $F = \Delta p / \Delta t = -45 / 0.002 = -22,500$ N. The negative sign shows the force was opposite to the initial direction — about 22,500 N!</details>

3. **Two identical balls undergo an elastic collision. Ball A has initial velocity v, Ball B is at rest. What are their final velocities?**
   <details><summary>Answer</summary>For equal masses in elastic collision: the moving ball stops, the stationary ball takes all the velocity. So Ball A final = 0, Ball B final = v. (This follows from solving the elastic collision equations with $m_1 = m_2$.)</details>

## Key Takeaways

- Momentum $\vec{p} = m\vec{v}$ is a vector — conserve it component-wise in each direction
- Impulse $\vec{J} = \vec{F}\Delta t = \Delta\vec{p}$ relates force, time, and momentum change
- In any collision: momentum is ALWAYS conserved (in an isolated system)
- Kinetic energy is only conserved in elastic collisions
- Perfectly inelastic collisions (objects stick together) lose the most kinetic energy
