# Results and Key Findings

This document summarizes the expected results and key findings from the Brain Connectome analysis.

---

## 1. Dataset Overview

### Human Connectome Project (HCP) Data

| Metric | Value |
|--------|-------|
| Total Subjects | ~1,058 |
| Brain Regions | 68 (Desikan-Killiany atlas) |
| Structural PCs | 60 |
| Functional PCs | 60 |
| Traits | 175+ behavioral/cognitive measures |

### Data Sources

- **Structural Connectome (SC)**: White matter fiber tract counts
- **Functional Connectome (FC)**: Resting-state fMRI correlations
- **TNPCA**: Tensor Network PCA coefficients (60 components each)

---

## 2. Sexual Dimorphism Analysis

### Methods

- **Statistical Test**: Welch's t-test (unequal variances)
- **Effect Size**: Cohen's d
- **Multiple Comparison Correction**: FDR (Benjamini-Hochberg)
- **Significance Threshold**: α = 0.05

### Expected Results

| Metric | Expected Value |
|--------|----------------|
| Significant Structural PCs | 20-30 (out of 60) |
| Largest Effect Size (Cohen's d) | ~0.5-0.8 |
| Most Significant Feature | Struct_PC1 or Struct_PC2 |

### Interpretation

Sexual dimorphism in brain connectivity is well-documented in the literature.
Key findings typically show:

1. **Total brain volume** differs significantly (males larger)
2. **PC1** often captures overall connectivity strength (correlated with volume)
3. **Higher-order PCs** may capture more nuanced structural differences

---

## 3. Machine Learning Classification

### Gender Classification

#### Methods

- **Algorithm**: Random Forest (500 trees)
- **Features**: 120 PCs (60 structural + 60 functional)
- **Train/Test Split**: 70/30 with stratification
- **Random Seed**: 42

#### Expected Results

| Metric | Expected Value |
|--------|----------------|
| Test Accuracy | 85-90% |
| Top Feature | Struct_PC1 (brain size proxy) |
| AUC-ROC | ~0.92-0.95 |

#### Feature Importance

Top biomarkers for gender classification typically include:

1. **Struct_PC1**: ~15-20% importance (total connectivity)
2. **FS_BrainSeg_Vol**: Brain volume (if included)
3. **Struct_PC2-5**: Secondary structural patterns

### Model Validation

The high accuracy is expected because:
- Brain volume is a strong predictor of sex
- Multiple PCs capture complementary information
- Large sample size provides robust estimates

**Caution**: When controlling for brain volume, accuracy drops to ~70-75%,
suggesting volume accounts for much of the predictive signal.

---

## 4. PCA Analysis

### Structural Connectome

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| PC1 | ~15-20% | 15-20% |
| PC1-10 | ~50-60% | 50-60% |
| PC1-60 | ~85-90% | 85-90% |

### Functional Connectome

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| PC1 | ~10-15% | 10-15% |
| PC1-10 | ~40-50% | 40-50% |
| PC1-60 | ~75-85% | 75-85% |

### VAE Comparison

The Variational Autoencoder typically achieves:
- **Reconstruction R²**: ~70-80%
- **Latent Dimensions**: 60 (matching PCA)

---

## 5. Validation Against Literature

### Expected Findings Consistent with Literature

1. **Sexual dimorphism in SC**: Males show higher total connectivity
   - Reference: Ingalhalikar et al. (2014) PNAS

2. **Functional connectivity patterns**: More variable than structural
   - Reference: Finn et al. (2015) Nature Neuroscience

3. **PC1 as global signal**: First PC captures overall connectivity strength
   - Reference: Zhu et al. (2019) NeuroImage (TNPCA paper)

---

## 6. Output Files

After running `python Runners/run_pipeline.py`, expect:

```
output/
├── dimorphism_results.csv    # Statistical test results
└── ml_results.csv            # Feature importance rankings
```

### dimorphism_results.csv

| Column | Description |
|--------|-------------|
| Feature | PC name (e.g., Struct_PC1) |
| T_Statistic | Welch's t-test statistic |
| P_Value | Raw p-value |
| Cohen_D | Effect size |
| Male_Mean | Mean value for males |
| Female_Mean | Mean value for females |
| P_Adjusted | FDR-corrected p-value |
| Significant | True if P_Adjusted < 0.05 |

### ml_results.csv

| Column | Description |
|--------|-------------|
| Feature | Feature name |
| Importance | Random Forest importance score |

---

## 7. Troubleshooting

### Low Classification Accuracy (<80%)

- Check for data loading issues
- Verify train/test split stratification
- Ensure no data leakage

### No Significant Dimorphism Results

- Verify data contains both genders
- Check for NaN values in features
- Confirm correct column names

### Pipeline Errors

- Ensure all data files are present
- Check Python version (3.9+)
- Verify dependencies installed
