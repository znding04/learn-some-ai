---
title: "Neuroimaging Analysis with Machine Learning"
level: intermediate
topic: ai-for-neuroscience
order: 8
---

# Neuroimaging Analysis with Machine Learning

## Overview

The standard analysis pipeline for fMRI has two broad approaches: mass-univariate analysis (find which brain regions respond to a condition) and multivariate pattern analysis (MVPA, also called searchlight analysis — decode information from distributed patterns). ML extends MVPA into a powerful framework for discovering structure in neuroimaging data.

Mass-univariate analysis fits a General Linear Model (GLM) to each voxel's timeseries independently, testing whether the regression coefficient for a condition is significantly different from zero. This tells you which individual voxels activate, but misses the fact that information is encoded in patterns across voxels. A voxel-by-voxel analysis might find no significant activation for "face vs. object" in any individual face-selective voxel, yet the pattern across all voxels clearly distinguishes faces from objects.

MVPA trains a classifier (SVM, logistic regression, neural network) to distinguish experimental conditions (e.g., "saw a face" vs. "saw an object") from the distributed pattern of brain activity across all voxels. The classifier's accuracy serves as an index of how much information is present in that pattern. The searchlight approach runs the classifier in a sphere around each voxel, producing a map of "decoding accuracy" across the brain — revealing which regions carry task-relevant information.

**Representational Similarity Analysis (RSA)** takes a different approach: it compares brain representations to model representations using similarity matrices. If the pattern similarity structure across conditions in the brain matches the pattern structure in a deep neural network, it suggests they use similar representations. RSA is particularly powerful because it avoids classifier training altogether and directly compares geometric structures.

**Feature extraction** is critical: raw voxel values are too high-dimensional and noisy. Common approaches include:
- **ROI-based**: Average activity within anatomically defined regions of interest
- **Time series features**: Mean, variance, slope over the experiment
- **Connectivity features**: Functional connectivity matrices (correlation between regional time series)
- **Decomposition**: PCA, ICA (Independent Component Analysis), dictionary learning

Modern neuroimaging ML also uses **nested cross-validation** to avoid inflated accuracy from information leakage: an outer loop splits data into train/test, while an inner loop selects features and hyperparameters. Failure to use nested CV is one of the most common pitfalls in neuroimaging ML papers.

## Key Concepts

- **MVPA (Multivariate Pattern Analysis)**: Using classifiers on distributed brain activity patterns to decode cognitive states
- **Searchlight**: A sphere moved across the brain, with a classifier trained at each location to find regions carrying information
- **Representational Similarity Analysis (RSA)**: Comparing brain and model representations via similarity matrices
- **Pattern similarity**: The correlation (or cosine similarity) between two brain activity patterns — used to measure how similarly two conditions are represented
- **Representational Dissimilarity Matrix (RDM)**: A matrix where element $(i,j)$ is 1 minus the similarity between condition $i$ and condition $j$
- **ICA (Independent Component Analysis)**: Decomposing fMRI data into spatially independent components — separates neural signals from artifacts
- **Feature selection**: Choosing which voxels/regions to include in a classifier; reduces overfitting
- **Nested cross-validation**: Cross-validation within cross-validation to avoid information leakage during hyperparameter tuning

## Code Examples

```python
"""
Searchlight MVPA analysis on fMRI data
Using nilearn for the analysis.
"""
from nilearn import datasets, image, maskers, decoding
import numpy as np

# Fetch an fMRI dataset (example from nilearn)
data = datasets.fetch_adhd(n=1)
fmri_filename = data.func[0]

# Load the fMRI image as a 4D timeseries
fmri_img = image.load_img(fmri_filename)

# Define a mask (whole brain or specific ROI)
masker = maskers.NiftiMasker(
    mask_img=None,  # will compute from data
    standardize='zscore_sample',
    detrend=True,
    high_pass=0.01,
    t_r=2.0
)
masker.fit(fmri_img)
mask = masker.mask_img_

# Create a simple condition labels vector for demonstration
# (In real analysis, these come from experimental design files)
n_timepoints = image.get_img_shape(fmri_img)[3]
conditions = np.random.choice(['condition_A', 'condition_B'], size=n_timepoints)

# Searchlight with an SVM classifier
searchlight = decoding.SearchLight(
    mask_img=mask,
    process_mask_img=mask,
    estimator='svc',
    n_jobs=-1,
    verbose=0
)
searchlight.fit(fmri_img, conditions)

# Result: map of decoding accuracy across the brain
result_map = searchlight.scores_
print(f"Max decoding accuracy: {result_map.max():.3f}")
print(f"Mean decoding accuracy: {result_map[result_map > 0].mean():.3f}")
```

This searchlight reveals which brain regions contain sufficient information to decode the experimental condition. The resulting accuracy map can be displayed on the brain surface.

## Further Reading

- [nilearn decoding tutorial](https://nilearn.github.io/dev/decoding.html)
- [PyMVPA documentation](https://pymvpa.org/)
- [Neurosynth (meta-analysis of fMRI)](https://neurosynth.org/)