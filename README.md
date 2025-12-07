# Connectopy

[![CI](https://github.com/Sean0418/connectopy/actions/workflows/ci.yml/badge.svg)](https://github.com/Sean0418/connectopy/actions/workflows/ci.yml)
[![Docker](https://github.com/Sean0418/connectopy/actions/workflows/docker.yml/badge.svg)](https://github.com/Sean0418/connectopy/actions/workflows/docker.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sean0418/connectopy/blob/main/notebooks/colab_demo.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A Python package for analyzing brain structural and functional connectomes from the Human Connectome Project (HCP).

## Features

- **Data Loading**: Load and merge HCP structural/functional connectome data with traits
- **Dimensionality Reduction**: PCA and VAE for connectome feature extraction
- **Statistical Analysis**: Sexual dimorphism analysis with effect sizes and FDR correction
- **Mediation Analysis**: Test brain network mediation of cognitive-alcohol relationships by sex
- **Machine Learning**: Multiple classifier options with unified interface
  - Random Forest, XGBoost, EBM (Explainable Boosting), SVM, Logistic Regression
  - Cross-validation with hyperparameter tuning (GridSearchCV)
  - Class imbalance handling (sample weights, SMOTE, undersampling)
  - Feature selection (SelectKBest)
  - Optimal threshold finding (F1-based)
  - Comprehensive metrics (AUC, balanced accuracy, precision, recall, F1)
- **Alcohol Classification**: Sex-stratified prediction of alcohol use disorder from brain + cognitive features
- **Visualization**: Publication-ready plots for connectome analysis (ROC curves, feature importance)
- **Reproducibility**: Docker container and automated pipelines

## One-Click Demo (No Setup Required)

Try the analysis instantly in Google Colab - no installation needed!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sean0418/connectopy/blob/main/notebooks/colab_demo.ipynb)

Just click the badge above and then **Runtime → Run all** to execute the entire analysis.

## Quick Start with Docker

The easiest way to run the analysis pipeline locally:

```bash
# Pull the latest image
docker pull ghcr.io/sean0418/connectopy:latest

# Run the pipeline (mount your data and output directories)
docker run -v /path/to/your/data:/app/data \
           -v /path/to/output:/app/output \
           ghcr.io/sean0418/connectopy:latest

# Run with options
docker run -v /path/to/data:/app/data \
           -v /path/to/output:/app/output \
           ghcr.io/sean0418/connectopy:latest --quick

# See all options
docker run ghcr.io/sean0418/connectopy:latest --help
```

## Installation (Development)

```bash
# Clone the repository
git clone https://github.com/Sean0418/connectopy.git
cd connectopy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

## Running the Pipeline

```bash
# Run the full analysis pipeline
python Runners/run_pipeline.py

# Quick mode (skip PCA, VAE, and plots)
python Runners/run_pipeline.py --quick

# Skip specific steps
python Runners/run_pipeline.py --skip-vae --skip-plots
```

### Pipeline Steps

| Step | Analysis | Output |
|------|----------|--------|
| 1 | Data Loading | Merged dataset |
| 2 | PCA Analysis | `pca_variance.csv`, `pca_scores.csv` |
| 3 | VAE Analysis | `vae_latent.csv`, `vae_training_history.csv` |
| 4 | Dimorphism Analysis | `dimorphism_results.csv` |
| 5 | ML Classification | `ml_results.csv`, `ebm_results.csv` |
| 6 | Mediation Analysis | `mediation_results.csv` |
| 7 | Visualization | `output/plots/*.png` |

Additional standalone analyses:
| Analysis | Runner | Output |
|----------|--------|--------|
| Alcohol Classification | `run_alcohol_analysis.py` | `output/alcohol_analysis/` |
| Mediation (Extended) | `run_mediation_hcp.py` | `output/mediation_*.csv` |

### Mediation Analysis on HCP Data

For a comprehensive sex-stratified mediation analysis:

```bash
# Run mediation analysis on HCP data
python Runners/run_mediation_hcp.py

# Outputs:
# - output/mediation_results_full.csv
# - output/mediation_sex_comparison.csv
# - output/mediation_results_significant.csv
# - output/mediation_*.png (visualizations)
# - output/MEDIATION_ANALYSIS_RESULTS.pdf (report)
```

### Alcohol Use Disorder Classification

Train Random Forest and Explainable Boosting Machine (EBM) classifiers to predict alcohol use disorder from brain connectome and cognitive features, stratified by sex:

```bash
# Run full analysis with all variants and models
python Runners/run_alcohol_analysis.py

# Run with specific variants only
python Runners/run_alcohol_analysis.py --variants tnpca

# Run Random Forest only
python Runners/run_alcohol_analysis.py --model-types rf

# Outputs:
# - output/alcohol_analysis/alcohol_classification_summary.csv
# - output/alcohol_analysis/models/ (trained model files)
# - output/alcohol_analysis/plots/roc/ (ROC curves)
# - output/alcohol_analysis/plots/importance/ (feature importance)
```

## Python API

```python
from connectopy import (
    ConnectomeDataLoader,
    DimorphismAnalysis,
    ConnectomeRandomForest,
    ConnectomeEBM,
)
from connectopy.models import get_cognitive_features, get_connectome_features

# Load data
loader = ConnectomeDataLoader("data/")
data = loader.load_merged_dataset()

# Analyze sexual dimorphism
analysis = DimorphismAnalysis(data)
features = [f"Struct_PC{i}" for i in range(1, 61)]
results = analysis.analyze(feature_columns=features)

# Get significant features
print(analysis.get_top_features(10))

# Train a classifier with CV and class imbalance handling
X = data[features].values
y = (data["Gender"] == "M").astype(int).values

clf = ConnectomeRandomForest()
metrics = clf.fit_with_cv(X, y, feature_names=features, handle_imbalance=True)
print(f"Test AUC: {metrics['test_auc']:.3f}")
print(f"Top biomarkers:\n{clf.get_top_features(5)}")

# Get feature sets for analysis
cog_features = get_cognitive_features(data)  # HCP cognitive measures
conn_features = get_connectome_features(data, "tnpca")  # TNPCA connectome features
```

### Mediation Analysis

Test whether brain networks mediate the relationship between cognitive traits and alcohol outcomes:

```python
from connectopy.analysis import SexStratifiedMediation

# Run sex-stratified mediation analysis
# Model: Cognitive → Brain Network → Alcohol Dependence
mediation = SexStratifiedMediation(n_bootstrap=1000)
result = mediation.fit(
    data=df,
    cognitive_col="FluidIntelligence",
    brain_col="SC_PC1",
    alcohol_col="AlcoholSeverity",
    sex_col="Gender",
)

print(f"Male indirect effect: {result.male.indirect_effect:.4f}")
print(f"Female indirect effect: {result.female.indirect_effect:.4f}")
print(f"Sex difference significant: {result.diff_significant}")
```

## Project Structure

```
connectopy/
├── src/
│   └── connectopy/             # Python package (src layout)
│       ├── data/               # Data loading (HCPLoader, preprocessing)
│       ├── analysis/           # PCA, VAE, dimorphism, mediation analysis
│       ├── models/             # ML classifiers (RF, XGBoost, EBM)
│       └── visualization/      # Plotting functions
├── Runners/                    # Pipeline execution scripts
├── tests/                      # Unit tests
├── docs/                       # Sphinx documentation
├── .github/workflows/          # CI/CD pipelines
├── data/                       # Data directory
│   ├── raw/                    # Raw HCP data files
│   └── processed/              # Generated datasets
├── output/                     # Analysis outputs
│   └── plots/                  # Generated visualizations
├── Dockerfile                  # Container definition
└── pyproject.toml              # Package configuration
```

## Data

The package expects HCP data in the following structure:

```
data/
├── raw/
│   ├── SC/                    # Structural Connectome .mat files
│   ├── FC/                    # Functional Connectome .mat files
│   ├── TNPCA_Result/          # Tensor Network PCA coefficients
│   └── traits/                # Subject trait CSV files
└── processed/                 # Generated datasets
```

**Data Access**: Raw data must be downloaded from [ConnectomeDB](https://db.humanconnectome.org/) after agreeing to HCP data usage terms.

## Development

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check .
ruff format .
mypy src/connectopy/
```

### Building Documentation

```bash
cd docs
make html

# View the docs:
# macOS
open _build/html/index.html

# Linux
xdg-open _build/html/index.html

# Windows
start _build/html/index.html
```

### Building Docker Image Locally

```bash
docker build -t connectopy .
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output connectopy
```

## Reproducibility Checklist

| Feature | Status |
|---------|--------|
| Python package with `pyproject.toml` | ✅ |
| 7-step automated analysis pipeline | ✅ |
| CI (linting, type checking, tests) | ✅ |
| Docker container (multi-arch: amd64 + arm64) | ✅ |
| GitHub Container Registry hosting | ✅ |
| Pre-commit hooks | ✅ |
| Sphinx documentation | ✅ |
| Reproducibility documentation | ✅ |

## CI/CD

This project uses GitHub Actions for:

- **CI** (on every push/PR): Linting, type checking, tests across Python 3.10-3.12
- **Docker** (on push to main): Builds and pushes multi-arch images to GitHub Container Registry

## Legacy R Code

The original R analysis is preserved in the `code/` directory. The `jasa-template` git tag marks the state before Python refactoring.

```bash
git checkout jasa-template
```

## Contributors

- Riley Harper
- Sean Shen
- Yinyu Yao

## License

MIT License - see LICENSE file for details.

## References

- Van Essen, D. C., et al. (2013). The WU-Minn Human Connectome Project: An overview. NeuroImage.
- Zhu, H., et al. (2019). Tensor Network Factorizations. NeuroImage.
