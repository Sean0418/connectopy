# Integrated HCP Datasets for Sexual Dimorphism Analysis

This folder contains two integrated datasets derived from the Human Connectome Project (HCP) S1200 release. Both datasets are aligned at the subject level and are designed for models that predict **sex** from brain connectivity and behavioral traits.

## Files

### 1. `sex_data_sc.csv`  — Structural only

- **Rows (subjects):** 1,065  
- **Columns:** 299  

Contains:
- 175 behavioral / demographic traits (renamed to their HCP variable names, e.g. `PMAT24_A_CR`, `ReadEng_AgeAdj`, `SSAGA_Alc_Hvy_Max_Drinks`, etc.).
- 60 structural TN-PCA components (`tn_sc_pc1`–`tn_sc_pc60`).
- 60 z-scored structural TN-PCA components (`tn_sc_pc1_z`–`tn_sc_pc60_z`).
- Demographics:
  - `Gender`: factor with levels `F` (female), `M` (male).
  - `Age`: ordered age-range factor with levels `22-25`, `26-30`, `31-35`, `36+`.

**Sex distribution:**

- Female: 575 (53.99%)  
- Male:   490 (46.01%)

**Age distribution (structural dataset):**

- `22-25`: 224  
- `26-30`: 467  
- `31-35`: 364  
- `36+`  : 10  

This dataset is appropriate for models that use only **structural connectivity** features (e.g., TN-PCA scores) plus traits and demographics.

---

### 2. `sex_data_sc_fc.csv`  — Structural + Functional

- **Rows (subjects):** 1,058  
- **Columns:** 419  

Contains:
- The same 175 behavioral / demographic traits as `sex_data_sc.csv`.
- 60 structural TN-PCA components (`tn_sc_pc1`–`tn_sc_pc60`) and their z-scored versions (`*_z`).
- 60 functional TN-PCA components (`tn_fc_pc1`–`tn_fc_pc60`) and their z-scored versions (`*_z`).
- Demographics:
  - `Gender`: factor with levels `F`, `M`.
  - `Age`: ordered age-range factor (`22-25`, `26-30`, `31-35`, `36+`).

**Sex distribution:**

- Female: 571 (53.97%)  
- Male:   487 (46.03%)

**Age distribution (SC+FC dataset):**

- `22-25`: 222  
- `26-30`: 464  
- `31-35`: 362  
- `36+`  : 10  

This dataset is intended for models that combine **structural and functional connectivity** features (e.g., EBM models using both SC and FC TN-PCA scores).

---

## Target and Features

- **Target variable for sexual dimorphism:**

  - `Gender` (or a derived binary target such as `sex_binary` if you create one):  
    - `F` = female  
    - `M` = male  

- **Main connectome-derived features:**

  - Structural TN-PCA (SC):
    - Raw: `tn_sc_pc1`, …, `tn_sc_pc60`  
    - Z-scored: `tn_sc_pc1_z`, …, `tn_sc_pc60_z`
  - Functional TN-PCA (FC) — only in `sex_data_sc_fc.csv`:
    - Raw: `tn_fc_pc1`, …, `tn_fc_pc60`  
    - Z-scored: `tn_fc_pc1_z`, …, `tn_fc_pc60_z`

- **Additional covariates:**

  - `Age` (ordered factor with 4 bins)
  - Behavioral traits (175 variables), e.g. cognitive scores, substance use, emotion, personality, etc.

---

## Notes on Class Imbalance

- **Sex:** The datasets are relatively balanced with slightly more females (~54%) than males (~46%). No strong corrective weighting is strictly necessary, but class weights can be used for robustness.
- **Age:** Age is stored as a categorical range. The majority of subjects fall into the `26-30` and `31-35` groups, with very few in `36+`. Any age-group based analysis should acknowledge this imbalance and treat age primarily as a covariate rather than a primary prediction target.

---

## Recommended Usage

- For **structural-only** models (e.g., sex classification from SC TN-PCA scores + traits): use `sex_data_sc.csv`.
- For **combined SC+FC** models and **shape plots with EBM**:
  - Use `sex_data_sc_fc.csv`.
  - Features: `tn_sc_pc*_z`, `tn_fc_pc*_z` (and optionally selected traits).
  - Target: `Gender` (convert to 0/1 if your modeling library requires numeric labels).

Example (Python / pseudocode):

```python
import pandas as pd

df = pd.read_csv("sex_data_sc_fc.csv")
X = df.filter(regex="^tn_(sc|fc)_pc\\d+_z$")  # all z-scored TN-PCA features
y = (df["Gender"] == "F").astype(int)        # 1 = Female, 0 = Male
