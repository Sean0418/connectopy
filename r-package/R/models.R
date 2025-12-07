#' Create a Random Forest Classifier
#'
#' Initialize a Random Forest classifier for connectome classification.
#'
#' @param n_estimators Number of trees (default: 500)
#' @param max_depth Maximum tree depth, or NULL for unlimited (default: NULL)
#' @param random_state Random seed for reproducibility (default: 42)
#'
#' @return A ConnectomeRandomForest object (Python reference)
#' @export
#'
#' @examples
#' \dontrun{
#' clf <- ConnectomeRandomForest(n_estimators = 500)
#' rf_fit(clf, X_train, y_train, feature_names = colnames(X_train))
#' predictions <- rf_predict(clf, X_test)
#' }
ConnectomeRandomForest <- function(n_estimators = 500L, max_depth = NULL,
                                    random_state = 42L) {
  bc <- get_connectopy()
  models <- reticulate::import("connectopy.models")
  models$ConnectomeRandomForest(
    n_estimators = as.integer(n_estimators),
    max_depth = max_depth,
    random_state = as.integer(random_state)
  )
}

#' Fit Random Forest Model
#'
#' Fit the Random Forest classifier.
#'
#' @param model A ConnectomeRandomForest object
#' @param X Numeric matrix of features (samples x features)
#' @param y Vector of labels
#' @param feature_names Character vector of feature names (optional)
#'
#' @return The fitted model (invisibly)
#' @export
rf_fit <- function(model, X, y, feature_names = NULL) {
  model$fit(X, y, feature_names = feature_names)
  invisible(model)
}

#' Predict with Random Forest
#'
#' Predict class labels.
#'
#' @param model A fitted ConnectomeRandomForest object
#' @param X Numeric matrix of features
#'
#' @return Vector of predicted labels
#' @export
rf_predict <- function(model, X) {
  model$predict(X)
}

#' Predict Probabilities with Random Forest
#'
#' Predict class probabilities.
#'
#' @param model A fitted ConnectomeRandomForest object
#' @param X Numeric matrix of features
#'
#' @return Matrix of class probabilities (samples x classes)
#' @export
rf_predict_proba <- function(model, X) {
  model$predict_proba(X)
}

#' Get Random Forest Feature Importance
#'
#' Get feature importance scores.
#'
#' @param model A fitted ConnectomeRandomForest object
#'
#' @return A data.frame with Feature and Importance columns
#' @export
rf_feature_importance <- function(model) {
  model$feature_importances_
}

#' Get Top Random Forest Features
#'
#' Get the top n most important features.
#'
#' @param model A fitted ConnectomeRandomForest object
#' @param n Number of top features (default: 10)
#'
#' @return A data.frame with top features and importance scores
#' @export
rf_top_features <- function(model, n = 10L) {
  model$get_top_features(n = as.integer(n))
}

#' Evaluate Random Forest Model
#'
#' Evaluate model on test set.
#'
#' @param model A fitted ConnectomeRandomForest object
#' @param X_test Test features matrix
#' @param y_test True labels
#'
#' @return A list with accuracy, confusion_matrix, and classification_report
#' @export
rf_evaluate <- function(model, X_test, y_test) {
  result <- model$evaluate(X_test, y_test)
  list(
    accuracy = result$accuracy,
    confusion_matrix = result$confusion_matrix,
    classification_report = result$classification_report
  )
}

# --- XGBoost Classifier ---

#' Create an XGBoost Classifier
#'
#' Initialize an XGBoost classifier for connectome classification.
#'
#' @param n_estimators Number of boosting rounds (default: 500)
#' @param max_depth Maximum tree depth (default: 4)
#' @param learning_rate Learning rate / eta (default: 0.05)
#' @param random_state Random seed for reproducibility (default: 42)
#'
#' @return A ConnectomeXGBoost object (Python reference)
#' @export
ConnectomeXGBoost <- function(n_estimators = 500L, max_depth = 4L,
                               learning_rate = 0.05, random_state = 42L) {
  models <- reticulate::import("connectopy.models")
  models$ConnectomeXGBoost(
    n_estimators = as.integer(n_estimators),
    max_depth = as.integer(max_depth),
    learning_rate = learning_rate,
    random_state = as.integer(random_state)
  )
}

#' Fit XGBoost Model
#'
#' @inheritParams rf_fit
#' @export
xgb_fit <- function(model, X, y, feature_names = NULL) {
  model$fit(X, y, feature_names = feature_names)
  invisible(model)
}

#' Predict with XGBoost
#'
#' @inheritParams rf_predict
#' @export
xgb_predict <- function(model, X) {
  model$predict(X)
}

#' Predict Probabilities with XGBoost
#'
#' @inheritParams rf_predict_proba
#' @export
xgb_predict_proba <- function(model, X) {
  model$predict_proba(X)
}

#' Get XGBoost Feature Importance
#'
#' @inheritParams rf_feature_importance
#' @export
xgb_feature_importance <- function(model) {
  model$feature_importances_
}

#' Get Top XGBoost Features
#'
#' @inheritParams rf_top_features
#' @export
xgb_top_features <- function(model, n = 10L) {
  model$get_top_features(n = as.integer(n))
}

#' Evaluate XGBoost Model
#'
#' @inheritParams rf_evaluate
#' @export
xgb_evaluate <- function(model, X_test, y_test) {
  result <- model$evaluate(X_test, y_test)
  list(
    accuracy = result$accuracy,
    confusion_matrix = result$confusion_matrix,
    classification_report = result$classification_report
  )
}

# --- EBM (Explainable Boosting Machine) ---

#' Create an EBM Classifier
#'
#' Initialize an Explainable Boosting Machine classifier.
#' Requires the interpret Python package.
#'
#' @param max_bins Maximum bins for continuous features (default: 256)
#' @param learning_rate Learning rate (default: 0.01)
#' @param max_leaves Maximum leaves per tree (default: 3)
#' @param interactions Number of interaction terms (default: 10)
#' @param random_state Random seed for reproducibility (default: 42)
#'
#' @return A ConnectomeEBM object (Python reference)
#' @export
ConnectomeEBM <- function(max_bins = 256L, learning_rate = 0.01,
                          max_leaves = 3L, interactions = 10L,
                          random_state = 42L) {
  models <- reticulate::import("connectopy.models")
  models$ConnectomeEBM(
    max_bins = as.integer(max_bins),
    learning_rate = learning_rate,
    max_leaves = as.integer(max_leaves),
    interactions = as.integer(interactions),
    random_state = as.integer(random_state)
  )
}

#' Fit EBM Model
#'
#' @inheritParams rf_fit
#' @export
ebm_fit <- function(model, X, y, feature_names = NULL) {
  model$fit(X, y, feature_names = feature_names)
  invisible(model)
}

#' Predict with EBM
#'
#' @inheritParams rf_predict
#' @export
ebm_predict <- function(model, X) {
  model$predict(X)
}

#' Predict Probabilities with EBM
#'
#' @inheritParams rf_predict_proba
#' @export
ebm_predict_proba <- function(model, X) {
  model$predict_proba(X)
}

#' Get Top EBM Features
#'
#' @inheritParams rf_top_features
#' @export
ebm_top_features <- function(model, n = 10L) {
  model$get_top_features(n = as.integer(n))
}

#' Evaluate EBM Model
#'
#' @inheritParams rf_evaluate
#' @export
ebm_evaluate <- function(model, X_test, y_test) {
  result <- model$evaluate(X_test, y_test)
  list(
    accuracy = result$accuracy,
    confusion_matrix = result$confusion_matrix,
    classification_report = result$classification_report
  )
}

# --- SVM Classifier ---

#' Create an SVM Classifier
#'
#' Initialize a Support Vector Machine classifier.
#'
#' @param C Regularization parameter (default: 1.0)
#' @param kernel Kernel type: "rbf", "linear", "poly" (default: "rbf")
#' @param gamma Kernel coefficient (default: "scale")
#' @param random_state Random seed for reproducibility (default: 42)
#'
#' @return A ConnectomeSVM object (Python reference)
#' @export
ConnectomeSVM <- function(C = 1.0, kernel = "rbf", gamma = "scale",
                          random_state = 42L) {
  models <- reticulate::import("connectopy.models")
  models$ConnectomeSVM(
    C = C,
    kernel = kernel,
    gamma = gamma,
    random_state = as.integer(random_state)
  )
}

#' Fit SVM Model
#'
#' @inheritParams rf_fit
#' @export
svm_fit <- function(model, X, y, feature_names = NULL) {
  model$fit(X, y, feature_names = feature_names)
  invisible(model)
}

#' Predict with SVM
#'
#' @inheritParams rf_predict
#' @export
svm_predict <- function(model, X) {
  model$predict(X)
}

#' Predict Probabilities with SVM
#'
#' @inheritParams rf_predict_proba
#' @export
svm_predict_proba <- function(model, X) {
  model$predict_proba(X)
}

#' Get Top SVM Features
#'
#' @inheritParams rf_top_features
#' @export
svm_top_features <- function(model, n = 10L) {
  model$get_top_features(n = as.integer(n))
}

#' Evaluate SVM Model
#'
#' @inheritParams rf_evaluate
#' @export
svm_evaluate <- function(model, X_test, y_test) {
  result <- model$evaluate(X_test, y_test)
  list(
    accuracy = result$accuracy,
    confusion_matrix = result$confusion_matrix,
    classification_report = result$classification_report
  )
}

# --- Logistic Regression Classifier ---

#' Create a Logistic Regression Classifier
#'
#' Initialize a regularized Logistic Regression classifier.
#'
#' @param C Inverse regularization strength (default: 1.0)
#' @param penalty Penalty type: "l1", "l2", "elasticnet" (default: "l2")
#' @param solver Solver: "lbfgs", "saga" (default: "lbfgs")
#' @param max_iter Maximum iterations (default: 1000)
#' @param random_state Random seed for reproducibility (default: 42)
#'
#' @return A ConnectomeLogistic object (Python reference)
#' @export
ConnectomeLogistic <- function(C = 1.0, penalty = "l2", solver = "lbfgs",
                               max_iter = 1000L, random_state = 42L) {
  models <- reticulate::import("connectopy.models")
  models$ConnectomeLogistic(
    C = C,
    penalty = penalty,
    solver = solver,
    max_iter = as.integer(max_iter),
    random_state = as.integer(random_state)
  )
}

#' Fit Logistic Regression Model
#'
#' @inheritParams rf_fit
#' @export
logistic_fit <- function(model, X, y, feature_names = NULL) {
  model$fit(X, y, feature_names = feature_names)
  invisible(model)
}

#' Predict with Logistic Regression
#'
#' @inheritParams rf_predict
#' @export
logistic_predict <- function(model, X) {
  model$predict(X)
}

#' Predict Probabilities with Logistic Regression
#'
#' @inheritParams rf_predict_proba
#' @export
logistic_predict_proba <- function(model, X) {
  model$predict_proba(X)
}

#' Get Logistic Regression Coefficients
#'
#' Get feature coefficients from the fitted model.
#'
#' @param model A fitted ConnectomeLogistic object
#'
#' @return A data.frame with Feature and Coefficient columns
#' @export
logistic_coefficients <- function(model) {
  model$get_coefficients()
}

#' Get Top Logistic Features
#'
#' @inheritParams rf_top_features
#' @export
logistic_top_features <- function(model, n = 10L) {
  model$get_top_features(n = as.integer(n))
}

#' Evaluate Logistic Regression Model
#'
#' @inheritParams rf_evaluate
#' @export
logistic_evaluate <- function(model, X_test, y_test) {
  result <- model$evaluate(X_test, y_test)
  list(
    accuracy = result$accuracy,
    confusion_matrix = result$confusion_matrix,
    classification_report = result$classification_report
  )
}

# --- fit_with_cv wrappers ---

#' Fit Random Forest with Cross-Validation
#'
#' Fit model using GridSearchCV with class imbalance handling.
#'
#' @param model A ConnectomeRandomForest object
#' @param X Numeric matrix of features
#' @param y Vector of labels
#' @param feature_names Character vector of feature names (optional)
#' @param test_size Proportion for test set (default: 0.2)
#' @param n_splits Number of CV folds (default: 5)
#' @param handle_imbalance Whether to handle class imbalance (default: TRUE)
#' @param param_grid Parameter grid for GridSearchCV (optional, uses Python defaults)
#'
#' @return A list with training metrics
#' @export
rf_fit_with_cv <- function(model, X, y, feature_names = NULL,
                           test_size = 0.2, n_splits = 5L,
                           handle_imbalance = TRUE, param_grid = NULL) {
  model$fit_with_cv(
    X, y,
    feature_names = feature_names,
    test_size = test_size,
    n_splits = as.integer(n_splits),
    handle_imbalance = handle_imbalance,
    param_grid = param_grid
  )
}

#' Fit EBM with Cross-Validation
#'
#' @inheritParams rf_fit_with_cv
#' @export
ebm_fit_with_cv <- function(model, X, y, feature_names = NULL,
                            test_size = 0.2, n_splits = 5L,
                            handle_imbalance = TRUE, param_grid = NULL) {
  model$fit_with_cv(
    X, y,
    feature_names = feature_names,
    test_size = test_size,
    n_splits = as.integer(n_splits),
    handle_imbalance = handle_imbalance,
    param_grid = param_grid
  )
}

#' Fit SVM with Cross-Validation
#'
#' @inheritParams rf_fit_with_cv
#' @param select_k_best Number of top features to select (default: 50)
#'
#' @export
svm_fit_with_cv <- function(model, X, y, feature_names = NULL,
                            test_size = 0.2, n_splits = 5L,
                            param_grid = NULL, select_k_best = 50L) {
  model$fit_with_cv(
    X, y,
    feature_names = feature_names,
    test_size = test_size,
    n_splits = as.integer(n_splits),
    param_grid = param_grid,
    select_k_best = as.integer(select_k_best)
  )
}

#' Fit Logistic Regression with Cross-Validation
#'
#' @inheritParams svm_fit_with_cv
#' @export
logistic_fit_with_cv <- function(model, X, y, feature_names = NULL,
                                  test_size = 0.2, n_splits = 5L,
                                  param_grid = NULL, select_k_best = 50L) {
  model$fit_with_cv(
    X, y,
    feature_names = feature_names,
    test_size = test_size,
    n_splits = as.integer(n_splits),
    param_grid = param_grid,
    select_k_best = as.integer(select_k_best)
  )
}
