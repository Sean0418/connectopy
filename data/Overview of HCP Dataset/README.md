# STOR 674 Final Project – HCP Connectomes and Traits

This repository contains code and configuration files for a STOR 674 course project using Human Connectome Project (HCP) data. We analyze how structural and functional brain connectivity relate to individual differences in cognition and heavy alcohol use.

## Data Overview

We work with a preprocessed subset of the HCP S1200 release:

- **Structural connectomes (SC)**  
  - File: `HCP_cortical_DesikanAtlas_SC.mat`  
  - Variables:
    - `all_id`: vector of length 1065 with subject IDs.
    - `hcp_sc_count`: array of size 68 × 68 × 1065, where each 68 × 68 matrix is a structural connectivity matrix (streamline counts) between Desikan–Killiany cortical regions.  
  - Summary:
    - 68 cortical regions (Desikan atlas).
    - 1065 subjects with valid structural connectomes.
    - ~26.5% of all entries are non-zero.
    - Upper-triangular edge density (edges > 0) has mean 0.537 (sd 0.033, min 0.279, max 0.626).

- **Functional connectomes (FC)**  
  - File: `HCP_cortical_DesikanAtlas_FC.mat`  
  - Variables:
    - `subj_list`: subject IDs.
    - `hcp_cortical_fc`: cell array; each non-empty cell is a 68 × 68 functional connectivity (correlation) matrix.  
  - We convert `hcp_cortical_fc` to a numeric array `fc_array` of size 68 × 68 × 1058 by:
    1. Dropping empty cells.
    2. Using the first non-empty matrix to infer the 68 × 68 shape.
    3. Stacking non-empty cells along a subject dimension.
  - Summary:
    - 1058 subjects with non-empty FC matrices.
    - Off-diagonal FC values range from about −0.58 to 0.99.
    - Mean off-diagonal FC = 0.302 (sd 0.225).

- **Tensor Network PCA (TN-PCA) coefficients**  
  - Structural:
    - File: `TNPCA_Coeff_HCP_Structural_Connectome.mat`
    - Variables: `PCA_Coeff` (TN-PCA scores), `sub_id` (subject IDs).
    - After squeezing, `PCA_Coeff` is 1065 × 60: 60 TN-PCA components per subject.
  - Functional:
    - File: `TNPCA_Coeff_HCP_Functional_Connectome.mat`
    - Variables: `PCA_Coeff`, `network_subject_ids`.
    - After squeezing, `PCA_Coeff` is 1058 × 60.
  - Both sets of TN-PCA scores are approximately standardized (mean ≈ 0, sd ≈ 0.031).

- **Behavioral and demographic traits (175-dimensional)**  
  - File: `HCP_175Traits.mat`
  - Variables:
    - `traits.175`: 175 × N matrix of behavioral traits.
    - `hcp.subj.id`: subject IDs corresponding to columns of `traits.175`.
  - Metadata:
    - File: `Details_175_Traits.xlsx`
    - Contains, for each of the 175 traits:
      - HCP variable name (column_header),
      - Category (Cognition, Substance Use, etc.),
      - Description.
  - In R, we transpose `traits.175` to construct a subject-level data frame (rows = subjects, columns = `trait1`…`trait175`) and then rename these columns using `Details_175_Traits.xlsx`.

## Integration Logic

1. **Connectome alignment**  
   - Structural IDs: `all_id`  
   - Functional IDs: filtered `subj_list_fc` corresponding to non-empty FC cells.  
   - TN-PCA IDs:
     - Structural: `sub_id`, matches `all_id` as a set.
     - Functional: `network_subject_ids`, matches `subj_list_fc` as a set.

2. **Export TN-PCA to CSV (MATLAB)**  
   - Structural:
     - `PCA_Coeff` → `TN_Structural_TNPCAScores_1065.csv`
     - Columns: `tn_sc_pc1`…`tn_sc_pc60`, `subject`
   - Functional:
     - `PCA_Coeff` → `TN_Functional_TNPCAScores_1058.csv`
     - Columns: `tn_fc_pc1`…`tn_fc_pc60`, `subject`

3. **R integration**  
   - Read TN-PCA CSVs into `tn_sc_df` and `tn_fc_df`.
   - Read `HCP_175Traits.mat` into `traits_long` (subject × 175 traits).
   - Merge:
     - `analysis_sc`: subjects with traits + structural TN-PCA.
     - `analysis_sc_fc`: subjects with traits + structural + functional TN-PCA.
   - Create derived variables:
     - `read_score` = `ReadEng_AgeAdj` (reading),
     - `heavy_max` = `SSAGA_Alc_Hvy_Max_Drinks` (heavy drinking),
     - group labels for “poor” vs “good” readers and “light” vs “heavy” drinkers.

## Repository Structure (conceptual)

- `data/` (not tracked): original HCP `.mat` files.  
- `derived/`:
  - `TN_Structural_TNPCAScores_1065.csv`
  - `TN_Functional_TNPCAScores_1058.csv`
  - `HCP_175Traits.mat`
  - `Details_175_Traits.xlsx`
- `code/`:
  - `summarize_hcp_connectome.m`: MATLAB script that checks shapes, densities, and TN-PCA IDs.
  - `analysis.Rmd`: main R Markdown notebook with data integration, EDA, and models.
- `figures/`: exported plots for the report.

> **Note:** The raw HCP data are not included in this repository due to data use restrictions. Only derived summaries and code are tracked here.
