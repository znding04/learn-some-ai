---
title: "Cosmological Simulations and AI Emulators"
difficulty: advanced
topic: ai-for-astronomy
order: 6
estimatedTime: "30 minutes"
summary: "Neural network emulators as fast surrogates for computationally expensive cosmological N-body simulations, covering the matter power spectrum and parameter inference."
---
# Cosmological Simulations and AI Emulators

Running a cosmological simulation from first principles is one of the most computationally expensive tasks in science. The IllustrisTNG simulation suite consumed roughly 10 million CPU-hours on the Hazel Hen supercomputer in Stuttgart. The Millennium simulation (Springel et al. 2005), a landmark N-body run with $2160^3$ particles, was state of the art for years -- and each run locked in fixed cosmological parameters, meaning that exploring parameter space required additional million-CPU-hour campaigns. Neural network emulators offer a different strategy: train a surrogate model on a grid of simulations, then use that model to predict outputs at arbitrary parameter values in milliseconds.

## N-Body Simulations and Structure Formation

Gravity drives the large-scale structure of the universe. Starting from nearly uniform initial conditions (the cosmic microwave background at $z \approx 1100$), small density fluctuations grow under gravity. Over-dense regions collapse into dark matter halos that host galaxies; under-dense regions expand into cosmic voids.

The governing equations for $N$ particles with positions $\mathbf{x}_i$ and velocities $\mathbf{v}_i$ in comoving coordinates are:

$$\ddot{\mathbf{x}}_i + 2H\dot{\mathbf{x}}_i = -\frac{1}{a^2}\nabla\phi$$

$$\nabla^2\phi = 4\pi G \bar{\rho} a^2 \delta$$

where $H = \dot{a}/a$ is the Hubble parameter, $a$ is the scale factor, $\phi$ is the gravitational potential, $\bar{\rho}$ is the mean matter density, and $\delta = (\rho - \bar{\rho})/\bar{\rho}$ is the density contrast. Modern codes (Gadget-4, AREPO, SWIFT) solve these equations with the PM (particle-mesh) + tree method, achieving $\mathcal{O}(N \log N)$ scaling.

## The Matter Power Spectrum

The primary statistical summary of large-scale structure is the matter power spectrum $P(k)$, defined by:

$$\langle \tilde{\delta}(\mathbf{k}) \tilde{\delta}^*(\mathbf{k}') \rangle = (2\pi)^3 P(k) \delta^{(3)}(\mathbf{k} - \mathbf{k}')$$

where $\tilde{\delta}(\mathbf{k})$ is the Fourier transform of the density field. The power spectrum encodes how much clustering exists at each spatial scale $\lambda \sim 2\pi/k$. Its shape depends sensitively on cosmological parameters:

- $\Omega_m$: total matter density fraction (shifts the turnover scale)
- $\sigma_8$: amplitude of matter fluctuations at 8 Mpc/h (overall normalization)
- $H_0$: Hubble constant (sets conversion between angles and distances)
- $n_s$: spectral index of primordial fluctuations (tilts the spectrum)
- $\Omega_b$: baryon fraction (controls baryon acoustic oscillation amplitude)

The dark matter halo mass function $n(M)$ -- the number density of halos per unit mass -- follows a nearly universal form (Press-Schechter 1974, Sheth-Tormen 2002) that also depends on these parameters.

## The Emulator Strategy

```mermaid
flowchart TD
    A[Cosmological Parameter Grid\ne.g. Latin Hypercube Sampling\nover Omega_m, sigma_8, H_0, n_s, Omega_b] --> B[Run N Simulations\ne.g. Quijote: 44,100 simulations]
    B --> C[Measure Summary Statistics\nP(k), halo mass function,\nbispectrum, void statistics]
    C --> D[Train Neural Network Emulator\nInput: theta, Output: P(k)]
    D --> E[Fast Prediction\n< 1 ms per evaluation]
    E --> F[Bayesian Inference\nMarkov Chain Monte Carlo\nor Nested Sampling]
    F --> G[Posterior over Cosmological\nParameters given observations]
```

The Quijote simulation suite (Villaescusa-Navarro et al. 2020, ApJS) provides 44,100 N-body simulations spanning a 5-parameter cosmological space -- specifically designed for emulator training and Fisher matrix estimation. The Latin Hypercube sampling ensures efficient coverage of the parameter space.

Notable emulators include:
- **CosmicEmu** (Heitmann et al. 2014): polynomial chaos expansion emulator for the nonlinear power spectrum
- **EuclidEmulator2** (Euclid Collaboration 2021): neural network emulator accurate to 1% for $k < 10\ h/\text{Mpc}$
- **BACCO** (Angulo et al. 2021): emulates baryonic effects on top of gravity-only simulations

## Code Example: Building a Neural Network Power Spectrum Emulator

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 1. Generate synthetic training data
#    In practice: load from Quijote or Boltzmann code (CAMB, CLASS) outputs
# ---------------------------------------------------------------------------

def linear_power_spectrum(k, omega_m=0.3, sigma8=0.8, ns=0.96, h=0.67):
    """
    Simplified linear P(k) approximation for illustration.
    Real emulators use CLASS/CAMB outputs for the training set.

    Uses the Eisenstein & Hu (1998) transfer function shape.
    """
    # Equality scale (Mpc/h)
    k_eq = 0.073 * omega_m * h  # approximate

    # Transfer function (Eisenstein & Hu fitting formula, simplified)
    q = k / (omega_m * h**2) * np.exp(omega_m + np.sqrt(2*h) * omega_m)
    T = np.log(1 + 2.34*q) / (2.34*q) * (
        1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
    )**(-0.25)

    # Primordial power spectrum
    k_pivot = 0.05  # Mpc^-1
    P_prim = k**ns

    # Normalize to sigma8
    P_lin = P_prim * T**2
    norm = sigma8**2 / (np.trapz(P_lin * (3 * (np.sin(8*k) - 8*k*np.cos(8*k))
                                           / (8*k)**3)**2 / (2*np.pi**2), k) + 1e-10)
    return norm * P_lin

np.random.seed(42)
N_train, N_test = 2000, 300
k_values = np.logspace(-2, 0.5, 50)  # k in h/Mpc

# Latin hypercube sampling over 5 parameters
# [Omega_m, sigma_8, H_0/100, n_s, Omega_b]
param_mins = np.array([0.1, 0.5, 0.5, 0.8, 0.03])
param_maxs = np.array([0.5, 1.2, 0.9, 1.2, 0.07])

def sample_params(N):
    # Simple random sampling (Latin hypercube in practice)
    return param_mins + np.random.rand(N, 5) * (param_maxs - param_mins)

params_train = sample_params(N_train)
params_test  = sample_params(N_test)

def compute_pk(params):
    omega_m, sigma8, h, ns, omega_b = params
    pk = linear_power_spectrum(k_values, omega_m=omega_m, sigma8=sigma8, ns=ns, h=h)
    return np.log10(pk + 1e-30)  # log-space for better regression

Pk_train = np.array([compute_pk(p) for p in params_train])
Pk_test  = np.array([compute_pk(p) for p in params_test])

# Normalize inputs
p_mean, p_std = params_train.mean(0), params_train.std(0)
Pk_mean, Pk_std = Pk_train.mean(0), Pk_train.std(0)

X_train = torch.FloatTensor((params_train - p_mean) / p_std)
Y_train = torch.FloatTensor((Pk_train - Pk_mean) / Pk_std)
X_test  = torch.FloatTensor((params_test  - p_mean) / p_std)
Y_test  = torch.FloatTensor((Pk_test  - Pk_mean) / Pk_std)

# ---------------------------------------------------------------------------
# 2. Define the emulator network
# ---------------------------------------------------------------------------

class PowerSpectrumEmulator(nn.Module):
    """
    Fully connected network: 5 cosmological parameters -> 50 P(k) values.
    Architecture similar to EuclidEmulator2.
    """
    def __init__(self, n_params=5, n_k=50, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_k),
        )

    def forward(self, theta):
        return self.net(theta)

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

model = PowerSpectrumEmulator()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

dataset = TensorDataset(X_train, Y_train)
loader  = DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(300):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

# ---------------------------------------------------------------------------
# 4. Evaluate accuracy
# ---------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    Pk_pred_norm = model(X_test).numpy()

Pk_pred = Pk_pred_norm * Pk_std + Pk_mean  # back to log10(P(k))
Pk_true = Y_test.numpy() * Pk_std + Pk_mean

# Percent error in P(k)
percent_err = np.abs(10**Pk_pred - 10**Pk_true) / (10**Pk_true) * 100
print(f"Median percent error in P(k): {np.median(percent_err):.2f}%")
print(f"95th percentile error:        {np.percentile(percent_err, 95):.2f}%")
print(f"Max error:                    {percent_err.max():.2f}%")

# Speed comparison
import time
theta_single = X_test[:1]
start = time.perf_counter()
for _ in range(10000):
    with torch.no_grad():
        _ = model(theta_single)
elapsed = (time.perf_counter() - start) / 10000 * 1000
print(f"\nEmulator evaluation time: {elapsed:.3f} ms per call")
print(f"Speedup vs simulation: ~10^7 x faster than a real N-body run")
```

## Uncertainty Quantification

A deterministic emulator gives point predictions, but Bayesian parameter inference requires knowing how much to trust those predictions. Three approaches are common:

1. **Ensemble methods**: train $M$ networks with different initializations; use variance across predictions as uncertainty estimate (Lakshminarayanan et al. 2017)
2. **Gaussian processes**: exact Bayesian framework, but scale poorly beyond $\sim 10^4$ training points; used by original CosmicEmu
3. **Bayesian neural networks**: place priors on weights, approximate posterior via variational inference; implemented in packages like Pyro and BayesFlow

For cosmological parameter inference, simulation-based inference (SBI) methods such as Sequential Neural Posterior Estimation (SNPE, Cranmer et al. 2020) bypass explicit likelihood evaluation entirely, using the emulator as a simulator and training a normalizing flow to approximate the posterior $p(\theta | \mathbf{d})$ directly.

## Key Concepts Summary

- **N-body simulations**: solve equations of motion for millions to billions of dark matter particles; codes include Gadget-4, AREPO, SWIFT
- **Matter power spectrum** $P(k)$: primary statistical summary of large-scale structure; sensitive to $\Omega_m$, $\sigma_8$, $H_0$, $n_s$, $\Omega_b$
- **Emulator pipeline**: train neural network on simulation grid (e.g., Quijote's 44,100 runs); achieve millisecond predictions vs. million-CPU-hour simulations
- **Uncertainty quantification**: ensembles, Gaussian processes, and Bayesian NNs quantify emulator uncertainty for reliable inference
- **Simulation-based inference**: SNPE and related methods perform full Bayesian parameter inference without explicit likelihoods

## Exercises

1. The Quijote simulations vary 5 cosmological parameters. If you wanted to sample the parameter space with a uniform grid using 10 points per dimension, how many simulations would you need? How does Latin Hypercube Sampling reduce this requirement while maintaining coverage?

2. The EuclidEmulator2 achieves 1% accuracy for $k < 10\ h/\text{Mpc}$. Stage-IV surveys like Euclid and LSST target sub-percent systematic control on the observed power spectrum. What implications does this have for emulator accuracy requirements?

3. Modify the `PowerSpectrumEmulator` to output uncertainty estimates using the ensemble approach: train 5 networks with different random seeds, then compute the mean and standard deviation of their predictions. How does the uncertainty vary across $k$?

4. The power spectrum emulator maps from a 5-dimensional parameter space to a 50-dimensional output. How would you adapt the architecture to emulate the halo mass function $n(M)$ instead? What additional challenges arise for quantities that are positive-definite and span many orders of magnitude?
