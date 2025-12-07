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
