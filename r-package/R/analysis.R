#' Create a Connectome PCA Model
#'
#' Initialize a PCA model for connectome analysis.
#'
#' @param n_components Number of principal components to retain (default: 60)
#' @param scale Logical, whether to standardize features before PCA (default: TRUE)
#'
#' @return A ConnectomePCA object (Python reference)
#' @export
#'
#' @examples
#' \dontrun{
#' pca <- ConnectomePCA(n_components = 60)
#' scores <- pca_fit_transform(pca, X)
#' }
ConnectomePCA <- function(n_components = 60L, scale = TRUE) {
  bc <- get_brain_connectome()
  bc$ConnectomePCA(n_components = as.integer(n_components), scale = scale)
}

#' Fit PCA Model
#'
#' Fit the PCA model to data.
#'
#' @param pca A ConnectomePCA object
#' @param X Numeric matrix of features (samples x features)
#'
#' @return The fitted PCA object (invisibly)
#' @export
pca_fit <- function(pca, X) {
  pca$fit(X)
  invisible(pca)
}

#' Transform Data with PCA
#'
#' Transform data to PCA space.
#'
#' @param pca A fitted ConnectomePCA object
#' @param X Numeric matrix of features (samples x features)
#'
#' @return Numeric matrix of PCA scores (samples x components)
#' @export
pca_transform <- function(pca, X) {
  pca$transform(X)
}

#' Fit and Transform with PCA
#'
#' Fit the PCA model and transform data in one step.
#'
#' @param pca A ConnectomePCA object
#' @param X Numeric matrix of features (samples x features)
#'
#' @return Numeric matrix of PCA scores (samples x components)
#' @export
pca_fit_transform <- function(pca, X) {
  pca$fit_transform(X)
}

#' Get PCA Variance Report
#'
#' Get a data frame with variance explained by each component.
#'
#' @param pca A fitted ConnectomePCA object
#'
#' @return A data.frame with Component, Variance_Explained, and Cumulative_Variance
#' @export
pca_variance_report <- function(pca) {
  pca$get_variance_report()
}

#' Get Total Variance Explained
#'
#' Get the total variance explained by all components.
#'
#' @param pca A fitted ConnectomePCA object
#'
#' @return Numeric, proportion of variance explained
#' @export
pca_total_variance <- function(pca) {
  pca$total_variance_explained_
}

# --- Dimorphism Analysis ---

#' Create a Dimorphism Analysis
#'
#' Initialize an analysis for sexual dimorphism in brain connectivity.
#'
#' @param data A data.frame containing features and a gender column
#' @param gender_column Name of the column containing gender labels (default: "Gender")
#' @param male_label Label for male subjects (default: "M")
#' @param female_label Label for female subjects (default: "F")
#'
#' @return A DimorphismAnalysis object (Python reference)
#' @export
#'
#' @examples
#' \dontrun{
#' analysis <- DimorphismAnalysis(data)
#' results <- dimorphism_analyze(analysis, feature_columns = paste0("PC", 1:60))
#' }
DimorphismAnalysis <- function(data, gender_column = "Gender",
                                male_label = "M", female_label = "F") {
  bc <- get_brain_connectome()
  bc$DimorphismAnalysis(
    data = data,
    gender_column = gender_column,
    male_label = male_label,
    female_label = female_label
  )
}

#' Run Dimorphism Analysis
#'
#' Perform statistical tests for sex differences in brain connectivity.
#'
#' @param analysis A DimorphismAnalysis object
#' @param feature_columns Character vector of feature column names to analyze.
#'   If NULL, uses all numeric columns.
#' @param alpha Significance threshold (default: 0.05)
#' @param correction_method Method for multiple comparison correction:
#'   "fdr_bh" (default), "bonferroni", or "none"
#'
#' @return A data.frame with test results including P_Value, Cohen_D, P_Adjusted,
#'   and Significant columns
#' @export
dimorphism_analyze <- function(analysis, feature_columns = NULL, alpha = 0.05,
                                correction_method = "fdr_bh") {
  analysis$analyze(
    feature_columns = feature_columns,
    alpha = alpha,
    correction_method = correction_method
  )
}

#' Get Top Dimorphic Features
#'
#' Get the top features showing sexual dimorphism.
#'
#' @param analysis A DimorphismAnalysis object (after calling dimorphism_analyze)
#' @param n Number of top features to return (default: 10)
#' @param by Criterion for ranking: "effect_size" (default) or "significance"
#'
#' @return A data.frame with top features
#' @export
dimorphism_top_features <- function(analysis, n = 10L, by = "effect_size") {
  analysis$get_top_features(n = as.integer(n), by = by)
}

#' Get Dimorphism Summary
#'
#' Get summary statistics of the dimorphism analysis.
#'
#' @param analysis A DimorphismAnalysis object (after calling dimorphism_analyze)
#'
#' @return A list with summary statistics
#' @export
dimorphism_summary <- function(analysis) {
  as.list(analysis$summary())
}
