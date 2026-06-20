---
title: "Photometric Redshifts and Large Sky Surveys"
difficulty: intermediate
topic: ai-for-astronomy
order: 9
estimatedTime: "30 minutes"
summary: "Photometric redshift estimation using machine learning as a scalable alternative to spectroscopic measurements for large sky surveys like LSST."
---

## Photometric Redshifts and Large Sky Surveys

## The Redshift Problem at Scale

The observable universe contains hundreds of billions of galaxies. Measuring the distance to each one requires knowing its redshift -- the fractional shift in wavelength caused by the expansion of spacetime between emission and observation:

$$z = \frac{\lambda_{\text{obs}} - \lambda_{\text{em}}}{\lambda_{\text{em}}} = \frac{\Delta\lambda}{\lambda_0}$$

Spectroscopic redshifts are gold standard. A spectrograph disperses a galaxy's light into a spectrum, revealing absorption and emission lines at known rest-frame wavelengths. Comparing observed positions to laboratory wavelengths yields $z$ to four or five decimal places. The problem: a single fiber spectrograph can observe perhaps a thousand objects per night. The Sloan Digital Sky Survey (SDSS) collected spectra for about 3.5 million galaxies over two decades. The Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) will image 20 billion objects. Spectroscopic follow-up of even 1% of that catalog is infeasible.

Photometric redshifts (photo-z) offer a practical alternative. Instead of dispersing light into thousands of wavelength bins, a photometric survey measures flux through a small number of broad filters -- typically five to ten covering ultraviolet through near-infrared. A galaxy at $z \approx 0.3$ shows its Balmer break shifted into the green band; at $z \approx 1.0$ the same break sits in the near-infrared. These characteristic color signatures allow statistical redshift estimation with photometry alone, trading precision for volume.

The tradeoff is real. Photometric redshifts carry uncertainties of $\sigma_z \sim 0.02(1+z)$ under good conditions, compared to $\sigma_z \sim 0.0001$ spectroscopically. Catastrophic outliers -- objects assigned entirely wrong redshifts due to color-color degeneracies -- occur in 1-5% of samples and bias weak lensing cosmology if not identified and removed.

## Template Fitting vs. Machine Learning

Two philosophical approaches to photo-z estimation have developed in parallel.

Template fitting methods (LePhare, BPZ, EAZY) build a library of spectral energy distribution templates spanning galaxy types from ellipticals to starbursts. For each galaxy, they compute the $\chi^2$ fit of every template redshifted through the survey filters to the observed magnitudes, producing a full probability density function $p(z|\mathbf{m})$ over redshift. Template methods generalize to new surveys without retraining and interpret failures physically, but they require accurate templates and can fail when real galaxies don't match the template library.

Machine learning methods treat photo-z as regression or density estimation. Given a training set of galaxies with known spectroscopic redshifts paired with their photometric measurements, supervised algorithms learn the mapping from colors to redshift. Key systems include:

- **ANNz2** (Sadeh, Abdalla & Lahav 2016): an ensemble of artificial neural networks that outputs full photo-z PDFs
- **TPZ** (Carrasco Kind & Brunner 2013): prediction trees with random forests, providing uncertainty estimates through the forest variance
- **METAPHOR** (Cavuoti et al. 2017): a machine learning pipeline with uncertainty quantification built in

Machine learning methods are typically more accurate than template fitting when training data is representative, but they fail silently when the test population differs from training -- a serious concern when spectroscopic surveys are brighter and bluer than the full photometric sample.

## Self-Organizing Maps for Redshift Calibration

A fundamental challenge is that spectroscopic training sets are magnitude-limited and thus biased. Objects too faint or too red to get spectra constitute the science sample but are absent from training. Self-organizing maps (SOMs) provide a nonlinear dimensionality reduction that maps the high-dimensional color space of a photometric catalog onto a two-dimensional grid, where nearby cells contain galaxies with similar colors.

By overlaying spectroscopic coverage onto the SOM, astronomers identify which regions of color space lack spectroscopic calibration. The SOMPZ method (Buchs et al. 2019, DES Collaboration) used this approach for Dark Energy Survey weak lensing, and it has become a standard tool for next-generation surveys. The SOM reveals not just where training data is missing but which photometric objects are genuinely well-calibrated versus poorly constrained.

## Key Surveys and Their Data Characteristics

**SDSS** (Sloan Digital Sky Survey): five-band ($ugriz$) imaging of 14,000 deg$^2$ to $r \approx 22.2$, with spectroscopy for 3.5 million galaxies. Foundational training set for photo-z algorithms.

**DES** (Dark Energy Survey): five-band ($grizY$) imaging to $i \approx 24$ over 5000 deg$^2$. Photo-z calibration for weak lensing cosmology using SOMs and cross-correlation techniques.

**Euclid**: ESA mission launched 2023, combining visible ($I_E$) imaging with near-infrared ($Y_J H$) photometry and slitless spectroscopy. Will observe 15,000 deg$^2$ to $z \sim 2$, requiring photo-z precision $\sigma_z/(1+z) < 0.05$ for 1.5 billion galaxies.

**Rubin/LSST**: six-band ($ugrizy$) imaging to $r \approx 27.5$ over 18,000 deg$^2$. Will produce the definitive photo-z catalog of the 2030s, with 20 billion detected objects.

**Gaia**: ESA astrometry mission providing parallaxes and proper motions for 1.5 billion stars to $G < 21$. While not a photo-z survey, Gaia transformed stellar astrophysics by enabling precise distances and 3D kinematics for the Milky Way. Gaia's BP/RP low-resolution spectra (Gaia DR3) also enable stellar parameter estimation via ML, a parallel problem to photo-z.

## Photo-z as Probability Density Functions

Modern photo-z estimation avoids point estimates. A galaxy's true redshift is uncertain, and that uncertainty is often non-Gaussian and multimodal -- a blue galaxy at $z=0.3$ may have nearly identical colors to a red galaxy at $z=1.0$. Representing this as a single number and an error bar discards information.

The community standard is now to provide a full $p(z|\mathbf{m})$ for each object. These PDFs are stored in files and propagated through downstream analyses. For weak lensing, the relevant quantity is the redshift distribution $n(z)$ of a tomographic bin, obtained by stacking individual PDFs:

$$n(z) = \sum_i p_i(z|\mathbf{m}_i) \cdot w_i$$

where $w_i$ are selection weights. Catastrophic outlier mitigation enters here: objects with broad, multimodal PDFs are downweighted or flagged. The COSMOS photometric redshift catalog (Laigle et al. 2016), with 30-band photometry, serves as a truth table for validating PDFs in smaller surveys.

## Photo-z Estimation Pipeline

```mermaid
flowchart TD
    A[Raw Survey Images] --> B[Source Extraction & Photometry]
    B --> C[Multi-band Flux Catalog]
    C --> D{Method Choice}
    D --> E[Template Fitting\nLePhare / EAZY]
    D --> F[ML Regression\nANNz2 / TPZ]
    E --> G[Photo-z PDF p(z|m)]
    F --> G
    G --> H[Catastrophic Outlier Flagging]
    H --> I[SOM-based Calibration]
    I --> J[Tomographic Bins n(z)]
    J --> K[Cosmological Analysis]
    subgraph Training
        L[Spectroscopic Survey\nSDSS / VIPERS] --> F
        M[SED Templates] --> E
    end
```

## Code Example: Photo-z Estimation with Random Forest

This example simulates five-band photometry for galaxies with known redshifts, trains a random forest regressor, and evaluates performance.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

rng = np.random.default_rng(42)

def simulate_galaxy_colors(n_galaxies=5000):
    """
    Simulate ugriz magnitudes with realistic color-redshift relations.
    Based on typical elliptical and late-type galaxy SEDs.
    """
    redshifts = rng.uniform(0.01, 1.0, n_galaxies)
    galaxy_types = rng.choice(['elliptical', 'spiral', 'starburst'],
                               size=n_galaxies, p=[0.3, 0.5, 0.2])

    mags = np.zeros((n_galaxies, 5))  # u, g, r, i, z

    for idx, (z, gtype) in enumerate(zip(redshifts, galaxy_types)):
        # r-band magnitude: brighter at lower z, scatter in luminosity
        r_mag = 18.0 + 4.0 * z + rng.normal(0, 0.5)

        if gtype == 'elliptical':
            # Red, old stellar populations; strong 4000 A break
            u_r = 2.0 + 1.5 * z + rng.normal(0, 0.1)
            g_r = 0.7 + 0.4 * z + rng.normal(0, 0.05)
            r_i = 0.35 + 0.3 * z + rng.normal(0, 0.05)
            i_z = 0.2 + 0.15 * z + rng.normal(0, 0.05)
        elif gtype == 'spiral':
            u_r = 1.2 + 0.8 * z + rng.normal(0, 0.15)
            g_r = 0.45 + 0.25 * z + rng.normal(0, 0.08)
            r_i = 0.2 + 0.2 * z + rng.normal(0, 0.06)
            i_z = 0.1 + 0.1 * z + rng.normal(0, 0.06)
        else:  # starburst
            u_r = 0.5 + 0.3 * z + rng.normal(0, 0.2)
            g_r = 0.2 + 0.15 * z + rng.normal(0, 0.1)
            r_i = 0.1 + 0.15 * z + rng.normal(0, 0.08)
            i_z = 0.05 + 0.08 * z + rng.normal(0, 0.07)

        mags[idx] = [r_mag + u_r, r_mag + g_r, r_mag, r_mag - r_i,
                     r_mag - r_i - i_z]

    # Add photometric noise (0.02-0.05 mag per band)
    noise = rng.normal(0, 0.03, mags.shape)
    mags += noise

    return mags, redshifts

# Generate dataset
mags, redshifts = simulate_galaxy_colors(n_galaxies=8000)

# Features: five magnitudes + four colors (u-g, g-r, r-i, i-z)
colors = np.diff(mags, axis=1)  # shape (n, 4)
features = np.hstack([mags, colors])

X_train, X_test, y_train, y_test = train_test_split(
    features, redshifts, test_size=0.2, random_state=42
)

# Train random forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
sigma_nmad = 1.4826 * np.median(
    np.abs((y_pred_rf - y_test) / (1 + y_test))
)
outlier_frac = np.mean(
    np.abs((y_pred_rf - y_test) / (1 + y_test)) > 0.15
)

print("=== Random Forest Photo-z Performance ===")
print(f"RMSE:              {rmse:.4f}")
print(f"sigma_NMAD:        {sigma_nmad:.4f}  (target < 0.05)")
print(f"Catastrophic rate: {outlier_frac:.3f}  (target < 0.05)")

# Use forest variance as uncertainty proxy
all_tree_preds = np.array([tree.predict(X_test)
                            for tree in rf.estimators_])
pred_std = all_tree_preds.std(axis=0)

print(f"\nMean prediction uncertainty: {pred_std.mean():.4f}")

# Feature importances
feature_names = ['u', 'g', 'r', 'i', 'z', 'u-g', 'g-r', 'r-i', 'i-z']
importances = rf.feature_importances_
ranked = sorted(zip(feature_names, importances),
                key=lambda x: x[1], reverse=True)

print("\nFeature importances:")
for name, imp in ranked:
    bar = '#' * int(imp * 100)
    print(f"  {name:5s}: {imp:.3f}  {bar}")
```

## Exercises

1. The Dark Energy Survey uses five bands ($grizY$) while SDSS uses five different bands ($ugriz$). How does the absence of $u$-band affect photo-z precision at $z < 0.5$? At what redshift does the Balmer break at 3646 A enter the $g$ filter?

2. A photo-z algorithm trained on SDSS spectroscopic galaxies (median $r \approx 17.7$) is applied to LSST objects reaching $r \approx 27.5$. What biases would you expect, and what observational strategy could mitigate them? (Consider the COSMOS deep field approach.)

3. Implement a simple SOM using `sklearn.neural_network` or `minisom` to cluster the simulated galaxy colors from the code example. Overlay the spectroscopic training coverage. Which regions of color space have fewer than 10 training objects per SOM cell?

4. The normalized median absolute deviation $\sigma_{\text{NMAD}} = 1.4826 \times \text{median}(|\Delta z| / (1 + z_{\text{spec}}))$ is preferred over RMSE for photo-z evaluation. Why does the factor of 1.4826 appear, and why is this statistic more robust to catastrophic outliers than standard deviation?
