# connectopy (R Package)

R interface to the `connectopy` Python package using `reticulate`.

## Installation

### 1. Install the Python package first

```bash
cd /path/to/connectopy
pip install -e .
```

### 2. Install the R package

```r
# Install reticulate if needed
install.packages("reticulate")
install.packages("devtools")

# Install connectopy from local source
devtools::install("r-package")
```

## Configuration

If you're using a virtual environment or conda:

```r
library(connectopy)

# Option 1: Use a virtualenv
configure_python(virtualenv = "~/.venv/connectopy")

# Option 2: Use conda
configure_python(condaenv = "connectopy")

# Option 3: Use specific Python
configure_python(python = "/usr/local/bin/python3")
```

## Quick Start

```r
library(connectopy)

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

### Advanced Training with Cross-Validation

```r
# Use fit_with_cv for GridSearchCV + class imbalance handling
clf <- ConnectomeRandomForest(n_estimators = 200L)
metrics <- rf_fit_with_cv(clf, X, y, 
                          feature_names = feature_cols,
                          handle_imbalance = TRUE)

cat("Test AUC:", metrics$test_auc, "\n")
cat("Test Balanced Accuracy:", metrics$test_bal_acc, "\n")

# Also works for EBM, SVM, Logistic
ebm <- ConnectomeEBM(max_leaves = 3L)
ebm_metrics <- ebm_fit_with_cv(ebm, X, y, feature_names = feature_cols)

svm <- ConnectomeSVM(kernel = "rbf")
svm_metrics <- svm_fit_with_cv(svm, X, y, feature_names = feature_cols)

logistic <- ConnectomeLogistic(penalty = "l2")
log_metrics <- logistic_fit_with_cv(logistic, X, y, feature_names = feature_cols)
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

**Random Forest**
- `ConnectomeRandomForest(n_estimators)` - Create RF classifier
- `rf_fit(model, X, y)` - Fit model
- `rf_fit_with_cv(model, X, y, ...)` - Fit with GridSearchCV
- `rf_predict(model, X)` - Predict
- `rf_evaluate(model, X_test, y_test)` - Evaluate
- `rf_top_features(model, n)` - Get important features

**XGBoost**
- `ConnectomeXGBoost(n_estimators)` - Create XGBoost classifier
- `xgb_fit`, `xgb_predict`, `xgb_evaluate`, `xgb_top_features`

**EBM (Explainable Boosting Machine)**
- `ConnectomeEBM(max_bins, learning_rate)` - Create EBM classifier
- `ebm_fit`, `ebm_fit_with_cv`, `ebm_predict`, `ebm_evaluate`, `ebm_top_features`

**SVM (Support Vector Machine)**
- `ConnectomeSVM(C, kernel, gamma)` - Create SVM classifier
- `svm_fit`, `svm_fit_with_cv`, `svm_predict`, `svm_evaluate`, `svm_top_features`

**Logistic Regression**
- `ConnectomeLogistic(C, penalty, solver)` - Create Logistic classifier
- `logistic_fit`, `logistic_fit_with_cv`, `logistic_predict`, `logistic_evaluate`
- `logistic_coefficients(model)` - Get feature coefficients
- `logistic_top_features(model, n)` - Get top features by coefficient

## How It Works

This R package is a thin wrapper around the Python `connectopy` package.
It uses `reticulate` to call Python functions, so:

1. **Any changes to the Python code automatically work in R** - no need to update the R package
2. **Data conversion is automatic** - R data.frames ↔ pandas DataFrames, matrices ↔ numpy arrays
3. **Same algorithms** - You get the exact same results as calling Python directly
