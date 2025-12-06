# brainconnectome (R Package)

R interface to the `brain-connectome` Python package using `reticulate`.

## Installation

### 1. Install the Python package first

```bash
cd /path/to/Brain-Connectome
pip install -e .
```

### 2. Install the R package

```r
# Install reticulate if needed
install.packages("reticulate")
install.packages("devtools")

# Install brainconnectome from local source
devtools::install("r-package")
```

## Configuration

If you're using a virtual environment or conda:

```r
library(brainconnectome)

# Option 1: Use a virtualenv
configure_python(virtualenv = "~/.venv/brain-connectome")

# Option 2: Use conda
configure_python(condaenv = "brain-connectome")

# Option 3: Use specific Python
configure_python(python = "/usr/local/bin/python3")
```

## Quick Start

```r
library(brainconnectome)

# Load data
loader <- ConnectomeDataLoader("data/")
data <- load_merged_dataset(loader)

# Analyze sexual dimorphism
analysis <- DimorphismAnalysis(data)
results <- dimorphism_analyze(analysis,
                               feature_columns = paste0("Struct_PC", 1:60))

# Get significant features
significant <- results[results$Significant, ]
print(significant)

# Get top features by effect size
top <- dimorphism_top_features(analysis, n = 10)
print(top)
```

## Machine Learning

```r
# Prepare data
feature_cols <- grep("^Struct_PC", names(data), value = TRUE)
X <- as.matrix(data[, feature_cols])
y <- ifelse(data$Gender == "M", 1L, 0L)

# Split data (simple example)
set.seed(42)
train_idx <- sample(nrow(X), 0.7 * nrow(X))
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Train Random Forest
clf <- ConnectomeRandomForest(n_estimators = 500L)
rf_fit(clf, X_train, y_train, feature_names = feature_cols)

# Evaluate
metrics <- rf_evaluate(clf, X_test, y_test)
cat("Accuracy:", metrics$accuracy, "\n")

# Feature importance
top_features <- rf_top_features(clf, n = 10)
print(top_features)
```

## PCA Analysis

```r
# Create PCA model
pca <- ConnectomePCA(n_components = 60L)

# Fit and transform
scores <- pca_fit_transform(pca, X)

# Check variance explained
cat("Total variance explained:", pca_total_variance(pca), "\n")

# Get variance report
report <- pca_variance_report(pca)
head(report)
```

## API Reference

### Data Loading
- `ConnectomeDataLoader(data_dir)` - Create data loader
- `load_merged_dataset(loader)` - Load all data merged
- `load_structural_connectome(loader)` - Load SC matrices
- `load_functional_connectome(loader)` - Load FC matrices
- `load_tnpca_structural(loader)` - Load structural TNPCA
- `load_tnpca_functional(loader)` - Load functional TNPCA
- `load_traits(loader)` - Load trait data

### Analysis
- `ConnectomePCA(n_components, scale)` - Create PCA model
- `pca_fit_transform(pca, X)` - Fit and transform
- `pca_variance_report(pca)` - Get variance explained
- `DimorphismAnalysis(data)` - Create dimorphism analysis
- `dimorphism_analyze(analysis, features)` - Run analysis
- `dimorphism_top_features(analysis, n)` - Get top features

### Models
- `ConnectomeRandomForest(n_estimators)` - Create RF classifier
- `rf_fit(model, X, y)` - Fit model
- `rf_predict(model, X)` - Predict
- `rf_evaluate(model, X_test, y_test)` - Evaluate
- `rf_top_features(model, n)` - Get important features
- `ConnectomeXGBoost(n_estimators)` - Create XGBoost classifier
- (Similar xgb_* functions)

## How It Works

This R package is a thin wrapper around the Python `brain-connectome` package.
It uses `reticulate` to call Python functions, so:

1. **Any changes to the Python code automatically work in R** - no need to update the R package
2. **Data conversion is automatic** - R data.frames ↔ pandas DataFrames, matrices ↔ numpy arrays
3. **Same algorithms** - You get the exact same results as calling Python directly
