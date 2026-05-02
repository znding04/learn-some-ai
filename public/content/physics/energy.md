# Kinetic Energy, Potential Energy, and Work

## Introduction

Energy is one of the most fundamental conserved quantities in physics. While forces tell us *how* motion changes, the **work-energy theorem** provides an alternative framework for analyzing motion — often simpler when dealing with varying forces or complex trajectories.

## Work

**Work** is the transfer of energy through force acting over a distance.

$$W = \vec{F} \cdot \vec{d} = Fd\cos\theta$$

where $\theta$ is the angle between the force and displacement vectors.

### Key Points
- Work is a **scalar** quantity (can be positive, negative, or zero)
- Only the component of force *along the displacement* does work
- Zero work when force ⊥ displacement (circular motion at constant speed)

### Units
$$[W] = \text{Joule (J)} = \text{N} \cdot \text{m}$$

## Kinetic Energy

The energy of motion:

$$K = \frac{1}{2}mv^2$$

### Work-Energy Theorem
The total work done on an object equals its change in kinetic energy:

$$W_{\text{net}} = \Delta K = K_f - K_i = \frac{1}{2}mv_f^2 - \frac{1}{2}mv_i^2$$

## Potential Energy

Energy stored due to an object's position in a **conservative force field**.

### Gravitational Potential Energy
For objects near Earth's surface:

$$U_g = mgh$$

where $h$ is the height above a chosen reference point.

More generally (for larger distances from Earth):

$$U_g = -\frac{GMm}{r}$$

### Elastic Potential Energy
For a spring compressed or extended by $x$:

$$U_s = \frac{1}{2}kx^2$$

where $k$ is the spring constant (units: N/m).

## Conservation of Mechanical Energy

In the absence of non-conservative forces (friction, air resistance):

$$E_{\text{mechanical}} = K + U = \text{constant}$$

$$K_i + U_i = K_f + U_f$$

This provides an elegant way to solve problems like pendulums, roller coasters, and falling objects.

### When Non-Conservative Forces Act
When friction or other dissipative forces are present:

$$W_{\text{nc}} = \Delta K + \Delta U = -(f_k \cdot d)$$

The work done by friction equals the change in mechanical energy.

## Power

Power is the rate of doing work (or transferring energy):

$$P = \frac{W}{t} = \frac{\Delta E}{\Delta t}$$

Units: **Watt (W)** = J/s

Also useful: $P = \vec{F} \cdot \vec{v}$ for constant force along velocity.

## Practice Problems

1. **A 2 kg ball is dropped from 5 m height. What is its speed just before hitting the ground? (ignore air resistance)**
   <details><summary>Answer</summary>Using conservation: $mgh = \frac{1}{2}mv^2$, so $v = \sqrt{2gh} = \sqrt{2 \times 9.8 \times 5} = \sqrt{98} \approx 9.9$ m/s</details>

2. **A spring with k = 200 N/m is compressed by 0.1 m. How much energy is stored? What speed can it give to a 0.5 kg mass?**
   <details><summary>Answer</summary>$U_s = \frac{1}{2}kx^2 = \frac{1}{2}(200)(0.1)^2 = 1$ J. This 1 J converts to $K = \frac{1}{2}mv^2 = 1$, so $v = \sqrt{2} \approx 1.41$ m/s</details>

3. **A 1000 kg car traveling at 30 m/s brakes to a stop. The brakes do 450,000 J of work. What was the car's initial kinetic energy, and how much additional work is needed?**
   <details><summary>Answer</summary>$K_i = \frac{1}{2}(1000)(30)^2 = 450,000$ J. The brakes exactly consumed all the kinetic energy — no additional work needed (the car stops).</details>

## Key Takeaways

- Work $W = \vec{F} \cdot \vec{d}$ transfers energy via force over distance
- Kinetic energy $K = \frac{1}{2}mv^2$ is always positive
- Potential energy depends on position: $U_g = mgh$ (gravitational), $U_s = \frac{1}{2}kx^2$ (spring)
- Conservation of mechanical energy applies when only conservative forces act
- The work-energy theorem always holds: $W_{\text{net}} = \Delta K$
