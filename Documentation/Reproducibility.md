# Reproducibility Guide

This document outlines all steps taken to ensure the Brain Connectome project is fully reproducible.

---

## 1. Quick Reproduction Guide

### Option A: One-Command Pipeline (Recommended)

```bash
# Clone repository
git clone https://github.com/Sean0418/Brain-Connectome.git
cd Brain-Connectome

# Run complete pipeline (auto-creates venv, installs deps)
python Runners/run_pipeline.py
```

### Option B: Manual Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package with dependencies
pip install -e ".[dev]"

# Run analysis
python Runners/run_pipeline.py
```

### Option C: Using R Interface

```r
# Install R package
devtools::install("r-package")

# Load and use
library(connectopy)
loader <- ConnectomeDataLoader("data/")
data <- load_merged_dataset(loader)
```

---

## 2. Environment Management

### 2.1 Virtual Environment Isolation

All dependencies are installed in an isolated Python virtual environment.
The `Runners/run_pipeline.py` script automatically creates and manages the
virtual environment - no manual setup required.

### 2.2 Pinned Dependencies

All Python packages are pinned to exact versions in `requirements.txt`:

```
numpy==1.26.4
pandas==2.2.0
scipy==1.12.0
scikit-learn==1.4.0
torch==2.2.0
xgboost==2.0.3
matplotlib==3.8.2
seaborn==0.13.2
h5py==3.10.0
```

This ensures anyone can recreate the exact same environment.

### 2.3 Environment Setup Module

`Runners/setup_environment.py` provides reusable environment management:

- Checks Python version requirements (3.9+)
- Creates virtual environment if not exists
- Installs dependencies from `requirements.txt`
- Can re-launch scripts in the virtual environment

---

## 3. Pipeline Automation

### 3.1 Single Entry Point

`Runners/run_pipeline.py` provides a single command to execute the entire pipeline:

```bash
python Runners/run_pipeline.py
```

### 3.2 Idempotent Execution

The pipeline checks for existing outputs before re-running each stage:

```python
if merged_data_path.exists():
    print("Loading cached merged dataset...")
    data = pd.read_csv(merged_data_path)
else:
    print("Loading and merging HCP data...")
    loader = ConnectomeDataLoader(str(data_dir))
    data = loader.load_merged_dataset()
```

### 3.3 Selective Execution

Skip specific steps if needed:

```bash
python Runners/run_pipeline.py --skip-dimorphism  # Skip statistical analysis
python Runners/run_pipeline.py --skip-ml          # Skip ML classification
```

---

## 4. Code Organization

### 4.1 Package Structure

```
Brain-Connectome/
├── connectopy/           # Python package
│   ├── data/                   # Data loading (loader.py, preprocessing.py)
│   ├── analysis/               # Analysis (pca.py, vae.py, dimorphism.py)
│   ├── models/                 # ML models (classifiers.py)
│   └── visualization/          # Plotting (plots.py)
├── r-package/                  # R interface via reticulate
├── Runners/                    # Pipeline automation
├── Documentation/              # Detailed documentation
├── tests/                      # Unit tests
└── docs/                       # Sphinx documentation
```

### 4.2 Centralized Configuration

All hyperparameters and defaults are defined in the module files with
clear documentation. Key defaults:

| Parameter | Default | Location |
|-----------|---------|----------|
| PCA components | 60 | `analysis/pca.py` |
| VAE latent dim | 60 | `analysis/vae.py` |
| RF n_estimators | 500 | `models/classifiers.py` |
| Random seed | 42 | Throughout |

### 4.3 Deterministic Execution

Random seeds are set for reproducibility:

```python
# In classifiers
clf = ConnectomeRandomForest(n_estimators=500, random_state=42)

# In train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

---

## 5. Testing

### 5.1 Unit Tests

Run the test suite:

```bash
pytest
```

Current coverage: 45% overall, with core modules at 89-98%.

### 5.2 Linting

Code quality is enforced with ruff:

```bash
ruff check .
ruff format .
```

### 5.3 Pre-commit Hooks

Pre-commit hooks automatically run on each commit:

```bash
pre-commit install  # One-time setup
```

---

## 6. Documentation

### 6.1 Code Documentation

All functions include NumPy-style docstrings:

```python
def analyze(self, feature_columns: list[str], alpha: float = 0.05):
    """
    Perform dimorphism analysis on specified features.

    Parameters
    ----------
    feature_columns : list of str
        List of feature column names to analyze.
    alpha : float, default=0.05
        Significance threshold.

    Returns
    -------
    results : DataFrame
        Results with P_Value, Cohen_D, P_Adjusted, Significant columns.
    """
```

### 6.2 Sphinx Documentation

Build the documentation:

```bash
cd docs
make html
open _build/html/index.html
```

### 6.3 Documentation Files

| Document | Description |
|----------|-------------|
| `README.md` | Project overview and quick start |
| `Documentation/Reproducibility.md` | This file |
| `Documentation/Results.md` | Key findings and expected results |
| `docs/` | Full Sphinx API documentation |
| `r-package/README.md` | R interface documentation |

---

## 7. Data Organization

### 7.1 Directory Structure

```
data/
├── raw/                    # Original data (immutable)
│   ├── SC/                 # Structural connectome .mat
│   ├── FC/                 # Functional connectome .mat
│   ├── TNPCA_Result/       # PCA coefficients
│   └── traits/             # Subject traits CSV
└── processed/              # Generated outputs
    └── full_data.csv       # Merged dataset
```

### 7.2 Data Access

Raw data must be downloaded from [ConnectomeDB](https://db.humanconnectome.org/)
after agreeing to HCP data usage terms.

---

## 8. Version Control

### 8.1 Git Tags

Important milestones are tagged:

- `jasa-template` - Original R analysis before Python refactor

### 8.2 Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable release |
| `develop` | Development |
| `sean` | Sean's contributions |
| `yinyu` | Yinyu's contributions |

---

## 9. Reproducibility Checklist

- [x] Source code published (GitHub)
- [x] Dependencies pinned (requirements.txt)
- [x] Environment setup automated (setup_environment.py)
- [x] Pipeline automated (run_pipeline.py)
- [x] Idempotent execution (skip existing outputs)
- [x] Random seeds set (deterministic results)
- [x] Unit tests (pytest)
- [x] Code linting (ruff)
- [x] Documentation (Sphinx + docstrings)
- [x] R interface (reticulate wrapper)
- [ ] Pre-computed outputs (GitHub releases) - TODO
- [ ] Docker container - TODO

---

## 10. Expected Results

See `Documentation/Results.md` for expected outputs and key findings.

### Quick Validation

After running the pipeline, you should see:

```
Pipeline Complete!
Outputs saved to: output/
  - dimorphism_results.csv
  - ml_results.csv
```

Expected metrics:
- **Significant dimorphic features**: ~20-30 PCs (FDR < 0.05)
- **Gender classification accuracy**: ~85-90%
