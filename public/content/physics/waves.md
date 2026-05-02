# Wave Properties, Sound, and Light

## Introduction

Waves are disturbances that transfer energy and momentum without transferring matter. They appear everywhere — from water ripples to sound to light itself. Understanding wave behavior is essential for physics, engineering, and understanding how we perceive sound and light.

## Wave Types

### Mechanical Waves
Require a medium (material) to propagate:
- Sound waves (air, water, solids)
- Water waves
- Seismic waves

### Electromagnetic Waves
Can propagate through vacuum:
- Light, radio waves, X-rays, gamma rays
- All travel at $c = 3 \times 10^8$ m/s in vacuum

## Wave Characteristics

| Property | Symbol | Description |
|----------|--------|-------------|
| Wavelength | $\lambda$ | Distance between successive peaks |
| Frequency | $f$ | Number of cycles per second (Hz) |
| Period | $T$ | Time for one complete cycle ($T = 1/f$) |
| Amplitude | $A$ | Maximum displacement from equilibrium |
| Wave speed | $v$ | How fast the wave travels ($v = f\lambda$) |

## Wave Equation

$$v = f\lambda$$

This fundamental relationship connects wavelength, frequency, and speed. For electromagnetic waves in vacuum: $c = f\lambda$.

## Wave Behavior

### Reflection
When a wave hits a boundary, it bounces back:
- Angle of incidence = Angle of reflection
- Mirror image of the original wave

### Refraction
When a wave passes from one medium to another, its speed changes:
$$n_1 v_1 = n_2 v_2$$
(where $n$ is the index of refraction)

The frequency stays constant; wavelength changes.

### Diffraction
Waves bend around obstacles and spread through openings:
- More pronounced when the opening is comparable to the wavelength
- Explains why sound can be heard around corners

### Interference
When two or more waves occupy the same space:

| Type | Condition | Result |
|------|-----------|--------|
| Constructive | Peaks align | Amplitude adds: $A_{\text{total}} = A_1 + A_2$ |
| Destructive | Peak meets trough | Amplitude cancels |

## Sound Waves

Sound is a **longitudinal** mechanical wave — particles oscillate parallel to the direction of propagation.

### Properties of Sound
- Requires a medium (cannot travel through vacuum)
- Speed in air: $\approx 343$ m/s at room temperature
- Speed depends on medium: $v_{\text{solid}} > v_{\text{liquid}} > v_{\text{gas}}$

### Intensity and Decibels
Sound intensity (power per area):
$$I = \frac{P}{A}$$

Human hearing spans an enormous range — we use decibels (dB) to compress this scale:

$$\beta = 10 \log_{10}\left(\frac{I}{I_0}\right)$$

where $I_0 = 10^{-12}$ W/m² is the threshold of hearing.

| Sound Level | Example |
|-------------|---------|
| 0 dB | Threshold of hearing |
| 60 dB | Normal conversation |
| 85 dB | Heavy traffic |
| 120 dB | Rock concert, threshold of pain |

### Doppler Effect
The apparent frequency shift when source and observer move relative to each other:

$$f' = f \frac{v \pm v_{\text{observer}}}{v \mp v_{\text{source}}}$$

## Light

Light is an **electromagnetic wave** — transverse, requiring no medium.

### The EM Spectrum

| Color | Wavelength Range |
|-------|-----------------|
| Violet | 380–450 nm |
| Blue | 450–495 nm |
| Green | 495–570 nm |
| Yellow | 570–590 nm |
| Orange | 590–620 nm |
| Red | 620–750 nm |

### Reflection and Refraction of Light

**Law of Reflection:** $\theta_i = \theta_r$

**Snell's Law:** $n_1 \sin\theta_1 = n_2 \sin\theta_2$

### Total Internal Reflection
When light tries to go from a denser to rarer medium at too steep an angle, it reflects completely:
$$\sin\theta_c = \frac{n_2}{n_1} \quad (\text{for } n_1 > n_2)$$

This principle powers fiber optic cables.

## Practice Problems

1. **A wave has frequency 256 Hz and wavelength 1.29 m. What is its speed?**
   <details><summary>Answer</summary>$v = f\lambda = 256 \times 1.29 \approx 330$ m/s (close to the speed of sound!)</details>

2. **An ambulance siren at 800 Hz approaches you at 30 m/s. The speed of sound is 343 m/s. What frequency do you hear?**
   <details><summary>Answer</summary>$f' = f \frac{v}{v - v_{\text{source}}} = 800 \times \frac{343}{343 - 30} = 800 \times 1.096 \approx 877$ Hz (noticeably higher)</details>

3. **Light goes from air (n=1) into glass (n=1.5). If the angle of incidence is 30°, what is the angle of refraction?**
   <details><summary>Answer</summary>Snell's law: $1 \cdot \sin 30° = 1.5 \cdot \sin \theta_2$, so $\sin\theta_2 = \frac{0.5}{1.5} = 0.333$, giving $\theta_2 \approx 19.5°$</details>

## Key Takeaways

- Wave speed: $v = f\lambda$ — all three quantities are related
- Mechanical waves need a medium; EM waves do not
- Sound: longitudinal wave, ~343 m/s in air, experiences Doppler shift
- Light: EM wave, $c = 3 \times 10^8$ m/s in vacuum, follows Snell's law at interfaces
- Reflection, refraction, diffraction, and interference are universal wave behaviors
