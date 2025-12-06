# Analysis Code

This directory contains the R analysis scripts for the Brain Connectome project.

## Files

| File | Description | Dependencies |
|------|-------------|--------------|
| `pca-vae.Rmd` | PCA and VAE dimensionality reduction on raw connectomes | Run first |
| `dataloader.Rmd` | Data loading and merging utilities | Requires `pca-vae.Rmd` outputs |
| `dimorphism.Rmd` | Sexual dimorphism analysis | Can run independently |
| `final_project2.Rmd` | Main analysis report with ML models | Can run independently |

## Execution Order

1. **`pca-vae.Rmd`** - Must run first to generate:
   - `../data/processed/raw_struct_pca.csv`
   - `../data/processed/raw_func_pca.csv`
   - `../data/processed/raw_pca_df.csv`
   - `../data/processed/vae_df.csv`

2. **`dataloader.Rmd`** - Merges all data into:
   - `../data/processed/full_data.csv`

3. **`dimorphism.Rmd`** & **`final_project2.Rmd`** - Can run after data is loaded

## Data Paths

All scripts reference data relative to the `code/` directory:
- Raw data: `../data/raw/`
- Processed data: `../data/processed/`
- Bibliography: `../manuscript/paperdti.bib`
