# Data Directory

This directory contains all data files for the Brain Connectome analysis.

## Structure

```
data/
├── raw/                              # Original data files (not generated)
│   ├── SC/                           # Structural Connectome
│   │   └── HCP_cortical_DesikanAtlas_SC.mat
│   ├── FC/                           # Functional Connectome
│   │   └── HCP_cortical_DesikanAtlas_FC.mat
│   ├── TNPCA_Result/                 # Tensor Network PCA results
│   │   ├── TNPCA_Coeff_HCP_Structural_Connectome.mat
│   │   └── TNPCA_Coeff_HCP_Functional_Connectome.mat
│   ├── traits/                       # Subject trait data
│   │   ├── table1_hcp.csv           # Demographic and cognitive traits
│   │   ├── table2_hcp.csv           # Additional traits
│   │   ├── age_gender_subjectid.mat
│   │   └── 175traits/               # Full 175 trait set
│   └── reference/                    # Reference papers
└── processed/                        # Generated data files
    ├── full_data.csv                # Merged dataset (from dataloader.Rmd)
    ├── raw_pca_df.csv               # Raw PCA scores (from pca-vae.Rmd)
    ├── vae_df.csv                   # VAE latent variables (from pca-vae.Rmd)
    ├── master_ml_dataset.csv        # ML-ready dataset
    └── ...
```

## Data Sources

### Structural Connectome (SC)
- 68×68 symmetric matrices per subject
- Represents white matter fiber tract counts between brain regions
- Parcellated using Desikan-Killiany atlas

### Functional Connectome (FC)
- 68×68 correlation matrices per subject
- From resting-state fMRI BOLD signal correlations

### TNPCA Results
- 60 principal components per subject
- From Tensor Network PCA decomposition

### Traits
- `table1_hcp.csv`: Age, Gender, Cognition scores (PMAT24, etc.)
- `table2_hcp.csv`: Additional behavioral measures
- `175traits/`: Comprehensive trait collection

## Data Access

Raw data must be downloaded from [ConnectomeDB](https://db.humanconnectome.org/) after agreeing to HCP data usage terms.

**Note**: Large data files (`.mat`, `.csv`) are excluded from git via `.gitignore`.

