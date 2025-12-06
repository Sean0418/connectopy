# Brain Connectome Analysis

Analysis of brain structural and functional connectomes from the Human Connectome Project (HCP) to study relationships between brain connectivity and human traits.

## Project Overview

This project analyzes the HCP dataset to study:
- Sexual dimorphism in brain structural connectomes
- Relationships between brain connectivity and cognitive traits
- Machine learning prediction of traits from connectome features

## Project Structure

```
Brain-Connectome/
├── code/                           # Analysis scripts
│   ├── pca-vae.Rmd                # PCA and VAE dimensionality reduction
│   ├── dataloader.Rmd             # Data loading and merging utilities
│   ├── dimorphism.Rmd             # Sexual dimorphism analysis
│   └── final_project2.Rmd         # Main analysis report
├── data/
│   ├── raw/                        # Raw data files
│   │   ├── SC/                     # Structural Connectome matrices
│   │   ├── FC/                     # Functional Connectome matrices
│   │   ├── TNPCA_Result/           # Tensor Network PCA coefficients
│   │   ├── traits/                 # Subject trait data
│   │   └── reference/              # Reference papers
│   └── processed/                  # Generated/processed datasets
│       ├── full_data.csv          # Merged dataset
│       ├── master_ml_dataset.csv  # ML-ready dataset
│       └── ...
├── manuscript/                     # Manuscript materials
│   └── paperdti.bib               # Bibliography
└── output/                         # Analysis outputs
```

## Data Sources

The data is from the Human Connectome Project (HCP) Young Adult study:
- **Structural Connectome (SC)**: 68×68 connectivity matrices representing white matter fiber connections
- **Functional Connectome (FC)**: 68×68 correlation matrices from resting-state fMRI
- **TNPCA Coefficients**: 60 principal components from Tensor Network PCA
- **Traits**: 175+ behavioral, cognitive, and demographic measures

## Analysis Workflow

1. **Run `pca-vae.Rmd` first**: Performs PCA and VAE dimensionality reduction on raw connectomes
2. **Run `dataloader.Rmd`**: Merges all data sources into a unified dataset
3. **Run `dimorphism.Rmd`**: Analyzes sexual dimorphism in brain connectivity
4. **Run `final_project2.Rmd`**: Full analysis with ML models and visualization

## Requirements

R packages required:
- `R.matlab` - for loading .mat files
- `dplyr`, `tidyverse` - data manipulation
- `ggplot2` - visualization
- `randomForest` - Random Forest models
- `xgboost` - Gradient boosting models
- `caret` - ML utilities
- `torch` - for VAE (optional)

## Data Access

To access the HCP data, you must:
1. Register at [ConnectomeDB](https://db.humanconnectome.org/)
2. Agree to the HCP data usage terms
3. Download the relevant data files

## References

- Van Essen, D. C., et al. (2013). The WU-Minn Human Connectome Project: An overview. NeuroImage.
- Zhu, H., et al. (2019). Tensor Network Factorizations. NeuroImage.

## Contributors

- Riley Harper
- Sean Shen
- Yinyu Yao
