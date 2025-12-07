"""Classification models for brain connectome analysis.

This module provides wrapper classes for Random Forest, XGBoost, and EBM classifiers
with additional functionality for feature importance analysis, hyperparameter tuning,
class imbalance handling, and connectome-specific reporting.

Key features:
- GridSearchCV for hyperparameter optimization
- Class imbalance handling via sample weights
- Feature selection utilities
- ROC curve generation and comprehensive metrics
- Model persistence with joblib
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier

    EBM_AVAILABLE = True
except ImportError:
    EBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_recall_curve


class ConnectomeRandomForest:
    """Random Forest classifier for brain connectome classification.

    This class wraps scikit-learn's RandomForestClassifier with additional
    functionality for feature importance analysis and reporting.

    Parameters
    ----------
    n_estimators : int, default=500
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of trees.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional arguments passed to RandomForestClassifier.

    Attributes
    ----------
    model : RandomForestClassifier
        Underlying sklearn model.
    feature_names : list or None
        Names of features used in training.
    feature_importances_ : DataFrame or None
        Feature importance scores after fitting.

    Examples
    --------
    >>> clf = ConnectomeRandomForest(n_estimators=500)
    >>> clf.fit(X_train, y_train, feature_names=feature_cols)
    >>> predictions = clf.predict(X_test)
    >>> top_features = clf.get_top_features(n=10)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the classifier."""
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self.feature_names: list[str] | None = None
        self.feature_importances_: pd.DataFrame | None = None
        self.classes_: NDArray | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
    ) -> ConnectomeRandomForest:
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names of features.

        Returns
        -------
        self : ConnectomeRandomForest
            Fitted classifier.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Store feature importances
        self.feature_importances_ = (
            pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Importance": self.model.feature_importances_,
                }
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        return self.model.predict_proba(X)

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top n most important features.

        Parameters
        ----------
        n : int, default=10
            Number of features to return.

        Returns
        -------
        top_features : DataFrame
            Top features with importance scores.

        Raises
        ------
        ValueError
            If model hasn't been fit.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fit first")
        return self.feature_importances_.head(n)

    def evaluate(
        self,
        X_test: NDArray[np.float64],
        y_test: NDArray,
    ) -> dict:
        """Evaluate model on test set.

        Parameters
        ----------
        X_test : ndarray of shape (n_samples, n_features)
            Test features.
        y_test : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        metrics : dict
            Dictionary containing accuracy, confusion matrix, and
            classification report.
        """
        y_pred = self.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

    def fit_with_cv(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
        test_size: float = 0.2,
        n_splits: int = 5,
        handle_imbalance: bool = True,
        param_grid: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """Fit the model with cross-validation and hyperparameter tuning.

        This method provides a complete training pipeline including:
        - Stratified train/test split
        - Class imbalance handling via sample weights
        - GridSearchCV for hyperparameter optimization
        - Comprehensive metrics computation

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names of features.
        test_size : float, default=0.2
            Proportion of data for testing.
        n_splits : int, default=5
            Number of CV folds.
        handle_imbalance : bool, default=True
            Whether to apply sample weights for class imbalance.
        param_grid : dict, optional
            Parameter grid for GridSearchCV. If None, uses defaults.

        Returns
        -------
        metrics : dict
            Dictionary containing train/test metrics, best parameters, etc.
        """
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Class imbalance handling
        sample_weight = None
        if handle_imbalance:
            class_counts = np.bincount(y_train.astype(int))
            n_total = len(y_train)
            w_neg = n_total / (2.0 * class_counts[0])
            w_pos = n_total / (2.0 * class_counts[1])
            sample_weight = np.where(y_train == 1, w_pos, w_neg)

        # Pipeline with imputation
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("rf", self.model),
            ]
        )

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                "rf__n_estimators": [300, 500],
                "rf__max_depth": [None, 10, 20],
                "rf__min_samples_leaf": [1, 5, 10],
                "rf__max_features": ["sqrt", "log2", 0.3],
            }

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        # Fit with sample weights
        fit_params = {}
        if sample_weight is not None:
            fit_params["rf__sample_weight"] = sample_weight

        grid.fit(X_train, y_train, **fit_params)

        # Store best model
        self.model = grid.best_estimator_.named_steps["rf"]
        self.classes_ = np.unique(y)
        self._best_params = grid.best_params_
        self._cv_score = grid.best_score_

        # Store feature importances
        self.feature_importances_ = (
            pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Importance": self.model.feature_importances_,
                }
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        # Compute comprehensive metrics
        y_train_proba = grid.best_estimator_.predict_proba(X_train)[:, 1]
        y_test_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        y_test_pred = (y_test_proba >= 0.5).astype(int)

        metrics = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "cv_best_auc": float(grid.best_score_),
            "train_auc": float(roc_auc_score(y_train, y_train_proba)),
            "test_auc": float(roc_auc_score(y_test, y_test_proba)),
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "train_bal_acc": float(balanced_accuracy_score(y_train, y_train_pred)),
            "test_bal_acc": float(balanced_accuracy_score(y_test, y_test_pred)),
            "train_precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
            "train_recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
            "train_f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
            "best_params": grid.best_params_,
        }

        # Store ROC data for plotting
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        self._roc_data = {"fpr": fpr, "tpr": tpr, "auc": metrics["test_auc"]}
        self._test_data = (X_test, y_test, y_test_proba)

        return metrics

    def get_roc_data(self) -> dict[str, NDArray]:
        """Get ROC curve data after fitting with CV.

        Returns
        -------
        roc_data : dict
            Dictionary with 'fpr', 'tpr', and 'auc' keys.
        """
        if not hasattr(self, "_roc_data"):
            raise ValueError("Must call fit_with_cv first")
        return self._roc_data


class ConnectomeXGBoost:
    """XGBoost classifier for brain connectome classification.

    This class wraps XGBoost with additional functionality for
    feature importance analysis and connectome-specific reporting.

    Parameters
    ----------
    n_estimators : int, default=500
        Number of boosting rounds.
    max_depth : int, default=4
        Maximum tree depth.
    learning_rate : float, default=0.05
        Learning rate (eta).
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional arguments passed to XGBClassifier.

    Attributes
    ----------
    model : XGBClassifier
        Underlying XGBoost model.
    feature_names : list or None
        Names of features used in training.
    feature_importances_ : DataFrame or None
        Feature importance scores after fitting.

    Raises
    ------
    ImportError
        If xgboost is not installed.

    Examples
    --------
    >>> clf = ConnectomeXGBoost(n_estimators=500, learning_rate=0.05)
    >>> clf.fit(X_train, y_train, feature_names=feature_cols)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for ConnectomeXGBoost")

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="error",
            **kwargs,
        )
        self.feature_names: list[str] | None = None
        self.feature_importances_: pd.DataFrame | None = None
        self.classes_: NDArray | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
    ) -> ConnectomeXGBoost:
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names of features.

        Returns
        -------
        self : ConnectomeXGBoost
            Fitted classifier.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Store feature importances
        self.feature_importances_ = (
            pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Importance": self.model.feature_importances_,
                }
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        return self.model.predict_proba(X)

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top n most important features.

        Parameters
        ----------
        n : int, default=10
            Number of features to return.

        Returns
        -------
        top_features : DataFrame
            Top features with importance scores.

        Raises
        ------
        ValueError
            If model hasn't been fit.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fit first")
        return self.feature_importances_.head(n)

    def evaluate(
        self,
        X_test: NDArray[np.float64],
        y_test: NDArray,
    ) -> dict:
        """Evaluate model on test set.

        Parameters
        ----------
        X_test : ndarray of shape (n_samples, n_features)
            Test features.
        y_test : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        metrics : dict
            Dictionary containing accuracy, confusion matrix, and
            classification report.
        """
        y_pred = self.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }


class ConnectomeEBM:
    """Explainable Boosting Machine for brain connectome classification.

    This class wraps InterpretML's ExplainableBoostingClassifier, providing
    a highly interpretable glass-box model for sex/phenotype classification.

    EBM is a tree-based, cyclic gradient boosting algorithm that provides
    both global and local explanations. It's particularly useful for
    understanding which brain connectivity features drive predictions.

    Parameters
    ----------
    max_bins : int, default=256
        Maximum number of bins per feature.
    max_interaction_bins : int, default=32
        Maximum number of bins for interaction terms.
    interactions : int, default=10
        Number of interaction terms to include.
    learning_rate : float, default=0.01
        Learning rate for boosting.
    max_leaves : int, default=3
        Maximum number of leaves per tree.
    min_samples_leaf : int, default=2
        Minimum samples per leaf.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional arguments passed to ExplainableBoostingClassifier.

    Attributes
    ----------
    model : ExplainableBoostingClassifier
        Underlying InterpretML model.
    feature_names : list or None
        Names of features used in training.
    feature_importances_ : DataFrame or None
        Feature importance scores after fitting.

    Raises
    ------
    ImportError
        If interpret is not installed.

    Examples
    --------
    >>> clf = ConnectomeEBM(learning_rate=0.01, max_leaves=3)
    >>> clf.fit(X_train, y_train, feature_names=feature_cols)
    >>> predictions = clf.predict(X_test)
    >>> explanation = clf.explain_global()
    """

    def __init__(
        self,
        max_bins: int = 256,
        max_interaction_bins: int = 32,
        interactions: int = 10,
        learning_rate: float = 0.01,
        max_leaves: int = 3,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the classifier."""
        if not EBM_AVAILABLE:
            raise ImportError(
                "interpret is required for ConnectomeEBM. Install with: pip install interpret"
            )

        self.random_state = random_state
        self.model = ExplainableBoostingClassifier(
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            interactions=interactions,
            learning_rate=learning_rate,
            max_leaves=max_leaves,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )
        self.feature_names: list[str] | None = None
        self.feature_importances_: pd.DataFrame | None = None
        self.classes_: NDArray | None = None
        self._best_params: dict[str, Any] | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
    ) -> ConnectomeEBM:
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names of features.

        Returns
        -------
        self : ConnectomeEBM
            Fitted classifier.
        """
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        self.model.fit(X, y)
        self.classes_ = np.array([0, 1])  # Binary classification

        # Extract feature importances from EBM
        # EBM stores term importances which we can use
        importances = []
        term_importances = self.model.term_importances()
        for i, _ in enumerate(self.feature_names):
            if i < len(term_importances):
                importances.append(term_importances[i])
            else:
                importances.append(0.0)

        self.feature_importances_ = (
            pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Importance": importances,
                }
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        return self.model.predict_proba(X)

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top n most important features.

        Parameters
        ----------
        n : int, default=10
            Number of features to return.

        Returns
        -------
        top_features : DataFrame
            Top features with importance scores.

        Raises
        ------
        ValueError
            If model hasn't been fit.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fit first")
        return self.feature_importances_.head(n)

    def evaluate(
        self,
        X_test: NDArray[np.float64],
        y_test: NDArray,
    ) -> dict:
        """Evaluate model on test set.

        Parameters
        ----------
        X_test : ndarray of shape (n_samples, n_features)
            Test features.
        y_test : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        metrics : dict
            Dictionary containing accuracy, confusion matrix, and
            classification report.
        """
        y_pred = self.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

    def explain_global(self) -> Any:
        """Get global explanation of the model.

        Returns the EBM's built-in global explanation which shows
        how each feature contributes to predictions on average.

        Returns
        -------
        explanation : EBMExplanation
            InterpretML explanation object that can be visualized.

        Raises
        ------
        ValueError
            If model hasn't been fit.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fit first")
        return self.model.explain_global()

    def explain_local(self, X: NDArray[np.float64]) -> Any:
        """Get local explanation for specific instances.

        Shows how each feature contributed to predictions for
        specific samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to explain.

        Returns
        -------
        explanation : EBMExplanation
            InterpretML explanation object that can be visualized.

        Raises
        ------
        ValueError
            If model hasn't been fit.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fit first")
        return self.model.explain_local(X)

    def fit_with_cv(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
        test_size: float = 0.2,
        n_splits: int = 5,
        handle_imbalance: bool = True,
        param_grid: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """Fit the model with cross-validation and hyperparameter tuning.

        This method provides a complete training pipeline including:
        - Stratified train/test split
        - Class imbalance handling via sample weights
        - GridSearchCV for hyperparameter optimization
        - Comprehensive metrics computation

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names of features.
        test_size : float, default=0.2
            Proportion of data for testing.
        n_splits : int, default=5
            Number of CV folds.
        handle_imbalance : bool, default=True
            Whether to apply sample weights for class imbalance.
        param_grid : dict, optional
            Parameter grid for GridSearchCV. If None, uses defaults.

        Returns
        -------
        metrics : dict
            Dictionary containing train/test metrics, best parameters, etc.
        """
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Class imbalance handling
        sample_weight = None
        if handle_imbalance:
            class_counts = np.bincount(y_train.astype(int))
            n_total = len(y_train)
            w_neg = n_total / (2.0 * class_counts[0])
            w_pos = n_total / (2.0 * class_counts[1])
            sample_weight = np.where(y_train == 1, w_pos, w_neg)

        # Default parameter grid with regularization for EBM
        if param_grid is None:
            param_grid = {
                "max_leaves": [2, 3],
                "min_samples_leaf": [50, 100],
                "learning_rate": [0.01],
                "outer_bags": [16],
                "max_bins": [32],
                "interactions": [0],  # Pure additive model
            }

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        # Fit with sample weights
        grid.fit(X_train, y_train, sample_weight=sample_weight)

        # Store best model
        self.model = grid.best_estimator_
        self.classes_ = np.array([0, 1])
        self._best_params = grid.best_params_
        self._cv_score = grid.best_score_

        # Extract feature importances from EBM
        importances = []
        term_importances = self.model.term_importances()
        for i, _ in enumerate(self.feature_names):
            if i < len(term_importances):
                importances.append(term_importances[i])
            else:
                importances.append(0.0)

        self.feature_importances_ = (
            pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Importance": importances,
                }
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        # Compute comprehensive metrics
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        y_test_pred = (y_test_proba >= 0.5).astype(int)

        metrics = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features_used": len(self.feature_names),
            "cv_best_auc": float(grid.best_score_),
            "train_auc": float(roc_auc_score(y_train, y_train_proba)),
            "test_auc": float(roc_auc_score(y_test, y_test_proba)),
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "train_bal_acc": float(balanced_accuracy_score(y_train, y_train_pred)),
            "test_bal_acc": float(balanced_accuracy_score(y_test, y_test_pred)),
            "train_precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
            "train_recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
            "train_f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
            "best_params": json.dumps(grid.best_params_),
        }

        # Store ROC data for plotting
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        self._roc_data = {"fpr": fpr, "tpr": tpr, "auc": metrics["test_auc"]}
        self._test_data = (X_test, y_test, y_test_proba)

        return metrics

    def get_roc_data(self) -> dict[str, NDArray]:
        """Get ROC curve data after fitting with CV.

        Returns
        -------
        roc_data : dict
            Dictionary with 'fpr', 'tpr', and 'auc' keys.
        """
        if not hasattr(self, "_roc_data"):
            raise ValueError("Must call fit_with_cv first")
        return self._roc_data


class ConnectomeSVM:
    """Support Vector Machine classifier for brain connectome classification.

    This class wraps scikit-learn's SVC with additional functionality for
    feature importance analysis (via permutation importance) and reporting.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    kernel : str, default='rbf'
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid').
    gamma : str or float, default='scale'
        Kernel coefficient.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional arguments passed to SVC.

    Attributes
    ----------
    model : SVC
        Underlying sklearn model.
    feature_names : list or None
        Names of features used in training.
    feature_importances_ : DataFrame or None
        Feature importance scores after fitting (permutation-based).

    Examples
    --------
    >>> clf = ConnectomeSVM(C=1.0, kernel='rbf')
    >>> clf.fit(X_train, y_train, feature_names=feature_cols)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the classifier."""
        self.random_state = random_state
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
            **kwargs,
        )
        self.feature_names: list[str] | None = None
        self.feature_importances_: pd.DataFrame | None = None
        self.classes_: NDArray | None = None
        self._scaler: StandardScaler | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
    ) -> ConnectomeSVM:
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names of features.

        Returns
        -------
        self : ConnectomeSVM
            Fitted classifier.
        """
        # Scale features (critical for SVM)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # For SVM, we'll compute feature importances later via permutation
        self.feature_importances_ = pd.DataFrame(
            {"Feature": self.feature_names, "Importance": np.zeros(len(self.feature_names))}
        )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predict class labels."""
        if self._scaler is None:
            raise ValueError("Model must be fit first")
        X_scaled = self._scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        if self._scaler is None:
            raise ValueError("Model must be fit first")
        X_scaled = self._scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top n most important features."""
        if self.feature_importances_ is None:
            raise ValueError("Model must be fit first")
        return self.feature_importances_.head(n)

    def evaluate(self, X_test: NDArray[np.float64], y_test: NDArray) -> dict:
        """Evaluate model on test set."""
        y_pred = self.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

    def fit_with_cv(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
        test_size: float = 0.2,
        n_splits: int = 5,
        handle_imbalance: bool = True,
        param_grid: dict[str, list] | None = None,
        select_k_best: int | None = 50,
        compare_resampling: bool = False,
        optimize_threshold: bool = True,
    ) -> dict[str, Any]:
        """Fit with cross-validation and hyperparameter tuning.

        Parameters
        ----------
        X : ndarray
            Training features.
        y : ndarray
            Target labels.
        feature_names : list, optional
            Names of features.
        test_size : float, default=0.2
            Proportion for test set.
        n_splits : int, default=5
            Number of CV folds.
        handle_imbalance : bool, default=True
            Whether to handle class imbalance.
        param_grid : dict, optional
            Custom parameter grid for GridSearchCV.
        select_k_best : int or None, default=50
            Number of top features to select. None disables selection.
        compare_resampling : bool, default=False
            If True and imblearn available, compare SMOTE/undersampling/weighting.
        optimize_threshold : bool, default=True
            If True, find optimal threshold based on F1 score.
        """
        from sklearn.inspection import permutation_importance

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Build pipeline steps
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

        if select_k_best is not None:
            steps.append(("selector", SelectKBest(f_classif, k=min(select_k_best, X.shape[1]))))

        steps.append(("svm", self.model))

        # Default parameter grid
        default_grid: dict[str, list] = {
            "svm__C": [0.1, 1, 10, 50],
            "svm__kernel": ["rbf", "linear", "poly"],
            "svm__gamma": ["scale", 0.01, 0.001],
        }
        if select_k_best is not None:
            default_grid["selector__k"] = [20, 50, min(100, X.shape[1])]

        grid_params = param_grid or default_grid
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Compare resampling strategies if requested and imblearn available
        best_grid = None
        best_score = -np.inf
        best_strategy = "weighting"

        if compare_resampling and IMBLEARN_AVAILABLE and handle_imbalance:
            # Calculate safe k_neighbors for SMOTE
            class_counts = np.bincount(y_train.astype(int))
            min_class = int(class_counts.min())
            smote_k = min(5, max(1, min_class - 1))

            strategies = {
                "weighting": Pipeline(steps),
                "smote": ImbPipeline(
                    steps[:-1]
                    + [
                        ("resampler", SMOTE(random_state=self.random_state, k_neighbors=smote_k)),
                        steps[-1],
                    ]
                ),
                "undersample": ImbPipeline(
                    steps[:-1]
                    + [("resampler", RandomUnderSampler(random_state=self.random_state)), steps[-1]]
                ),
            }

            for name, pipe in strategies.items():
                grid = GridSearchCV(
                    pipe, grid_params, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
                )
                grid.fit(X_train, y_train)
                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_grid = grid
                    best_strategy = name
        else:
            pipe = Pipeline(steps)
            best_grid = GridSearchCV(
                pipe, grid_params, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
            )
            best_grid.fit(X_train, y_train)
            best_score = best_grid.best_score_

        grid = best_grid
        assert grid is not None  # Always set by one of the branches above

        # Store best model
        self.model = grid.best_estimator_.named_steps["svm"]
        self._scaler = grid.best_estimator_.named_steps["scaler"]
        self.classes_ = np.unique(y)
        self._best_params = grid.best_params_
        self._cv_score = grid.best_score_

        # Compute permutation importance
        perm = permutation_importance(
            grid.best_estimator_,
            X_test,
            y_test,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_importances_ = (
            pd.DataFrame({"Feature": self.feature_names, "Importance": perm.importances_mean})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        # Get predictions
        y_train_proba = grid.best_estimator_.predict_proba(X_train)[:, 1]
        y_test_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]

        # Optimize threshold if requested
        threshold = 0.5
        if optimize_threshold:
            precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_scores = 2 * precision * recall / (precision + recall)
                f1_scores = np.nan_to_num(f1_scores, nan=0.0)
            if len(thresholds) > 0:
                best_idx = np.argmax(f1_scores[:-1])  # Last element is for threshold=1
                threshold = float(thresholds[best_idx])

        y_train_pred = (y_train_proba >= threshold).astype(int)
        y_test_pred = (y_test_proba >= threshold).astype(int)

        metrics = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "cv_best_auc": float(best_score),
            "train_auc": float(roc_auc_score(y_train, y_train_proba)),
            "test_auc": float(roc_auc_score(y_test, y_test_proba)),
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "train_bal_acc": float(balanced_accuracy_score(y_train, y_train_pred)),
            "test_bal_acc": float(balanced_accuracy_score(y_test, y_test_pred)),
            "train_precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
            "train_recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
            "train_f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
            "best_params": grid.best_params_,
            "resampling_strategy": best_strategy,
            "optimal_threshold": threshold,
        }

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        self._roc_data = {"fpr": fpr, "tpr": tpr, "auc": metrics["test_auc"]}
        self._test_data = (X_test, y_test, y_test_proba)

        return metrics

    def get_roc_data(self) -> dict[str, NDArray]:
        """Get ROC curve data after fitting with CV."""
        if not hasattr(self, "_roc_data"):
            raise ValueError("Must call fit_with_cv first")
        return self._roc_data


class ConnectomeLogistic:
    """Logistic Regression classifier for brain connectome classification.

    This class wraps scikit-learn's LogisticRegression with L1/L2/ElasticNet
    regularization for feature selection and interpretability.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength.
    penalty : str, default='l2'
        Regularization type ('l1', 'l2', 'elasticnet', None).
    solver : str, default='lbfgs'
        Optimization algorithm.
    max_iter : int, default=1000
        Maximum iterations for convergence.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional arguments passed to LogisticRegression.

    Attributes
    ----------
    model : LogisticRegression
        Underlying sklearn model.
    feature_names : list or None
        Names of features used in training.
    feature_importances_ : DataFrame or None
        Feature importance scores (absolute coefficients).

    Examples
    --------
    >>> clf = ConnectomeLogistic(C=1.0, penalty='l1', solver='saga')
    >>> clf.fit(X_train, y_train, feature_names=feature_cols)
    >>> predictions = clf.predict(X_test)
    >>> # L1 regularization provides sparse coefficients for feature selection
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the classifier."""
        self.random_state = random_state
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
            **kwargs,
        )
        self.feature_names: list[str] | None = None
        self.feature_importances_: pd.DataFrame | None = None
        self.classes_: NDArray | None = None
        self._scaler: StandardScaler | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
    ) -> ConnectomeLogistic:
        """Fit the model."""
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Use absolute coefficients as importance
        coefs = np.abs(self.model.coef_[0])
        self.feature_importances_ = (
            pd.DataFrame({"Feature": self.feature_names, "Importance": coefs})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predict class labels."""
        if self._scaler is None:
            raise ValueError("Model must be fit first")
        X_scaled = self._scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        if self._scaler is None:
            raise ValueError("Model must be fit first")
        X_scaled = self._scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top n most important features."""
        if self.feature_importances_ is None:
            raise ValueError("Model must be fit first")
        return self.feature_importances_.head(n)

    def get_coefficients(self) -> pd.DataFrame:
        """Get feature coefficients with signs (for interpretability).

        Returns
        -------
        coef_df : DataFrame
            DataFrame with Feature, Coefficient, and AbsCoef columns.
        """
        if self.feature_names is None:
            raise ValueError("Model must be fit first")
        coefs = self.model.coef_[0]
        return (
            pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Coefficient": coefs,
                    "AbsCoef": np.abs(coefs),
                }
            )
            .sort_values("AbsCoef", ascending=False)
            .reset_index(drop=True)
        )

    def evaluate(self, X_test: NDArray[np.float64], y_test: NDArray) -> dict:
        """Evaluate model on test set."""
        y_pred = self.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

    def fit_with_cv(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: list[str] | None = None,
        test_size: float = 0.2,
        n_splits: int = 5,
        handle_imbalance: bool = True,
        param_grid: dict[str, list] | list[dict[str, Any]] | None = None,
        select_k_best: int | None = 50,
        compare_resampling: bool = False,
        optimize_threshold: bool = True,
    ) -> dict[str, Any]:
        """Fit with cross-validation and hyperparameter tuning.

        Parameters
        ----------
        X : ndarray
            Training features.
        y : ndarray
            Target labels.
        feature_names : list, optional
            Names of features.
        test_size : float, default=0.2
            Proportion for test set.
        n_splits : int, default=5
            Number of CV folds.
        handle_imbalance : bool, default=True
            Whether to handle class imbalance.
        param_grid : dict or list, optional
            Custom parameter grid for GridSearchCV.
        select_k_best : int or None, default=50
            Number of top features to select. None disables selection.
        compare_resampling : bool, default=False
            If True and imblearn available, compare SMOTE/undersampling/weighting.
        optimize_threshold : bool, default=True
            If True, find optimal threshold based on F1 score.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Build pipeline steps
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

        if select_k_best is not None:
            steps.append(("selector", SelectKBest(f_classif, k=min(select_k_best, X.shape[1]))))

        steps.append(("logistic", self.model))

        # Default parameter grid with L1, L2, and ElasticNet
        if param_grid is None:
            param_grid = [
                {
                    "logistic__C": [0.01, 0.1, 1, 10],
                    "logistic__penalty": ["l2"],
                    "logistic__solver": ["lbfgs"],
                },
                {
                    "logistic__C": [0.01, 0.1, 1, 10],
                    "logistic__penalty": ["l1"],
                    "logistic__solver": ["saga"],
                },
                {
                    "logistic__C": [0.01, 0.1, 1, 10],
                    "logistic__penalty": ["elasticnet"],
                    "logistic__solver": ["saga"],
                    "logistic__l1_ratio": [0.3, 0.5, 0.7],
                },
            ]
            if select_k_best is not None:
                for pg in param_grid:
                    pg["selector__k"] = [20, 50, min(100, X.shape[1])]

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Compare resampling strategies if requested and imblearn available
        best_grid = None
        best_score = -np.inf
        best_strategy = "weighting"

        if compare_resampling and IMBLEARN_AVAILABLE and handle_imbalance:
            class_counts = np.bincount(y_train.astype(int))
            min_class = int(class_counts.min())
            smote_k = min(5, max(1, min_class - 1))

            strategies = {
                "weighting": Pipeline(steps),
                "smote": ImbPipeline(
                    steps[:-1]
                    + [
                        ("resampler", SMOTE(random_state=self.random_state, k_neighbors=smote_k)),
                        steps[-1],
                    ]
                ),
                "undersample": ImbPipeline(
                    steps[:-1]
                    + [("resampler", RandomUnderSampler(random_state=self.random_state)), steps[-1]]
                ),
            }

            for name, pipe in strategies.items():
                grid = GridSearchCV(
                    pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
                )
                grid.fit(X_train, y_train)
                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_grid = grid
                    best_strategy = name
        else:
            pipe = Pipeline(steps)
            best_grid = GridSearchCV(
                pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
            )
            best_grid.fit(X_train, y_train)
            best_score = best_grid.best_score_

        grid = best_grid
        assert grid is not None  # Always set by one of the branches above

        self.model = grid.best_estimator_.named_steps["logistic"]
        self._scaler = grid.best_estimator_.named_steps["scaler"]
        self.classes_ = np.unique(y)
        self._best_params = grid.best_params_
        self._cv_score = grid.best_score_

        # Use absolute coefficients as importance
        coefs = np.abs(self.model.coef_[0])

        # Handle feature selection - get selected feature names
        if "selector" in grid.best_estimator_.named_steps:
            selector = grid.best_estimator_.named_steps["selector"]
            selected_mask = selector.get_support()
            selected_features = [f for f, m in zip(self.feature_names, selected_mask) if m]
        else:
            selected_features = self.feature_names

        self.feature_importances_ = (
            pd.DataFrame({"Feature": selected_features, "Importance": coefs})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        y_train_proba = grid.best_estimator_.predict_proba(X_train)[:, 1]
        y_test_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]

        # Optimize threshold if requested
        threshold = 0.5
        if optimize_threshold:
            precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_scores = 2 * precision * recall / (precision + recall)
                f1_scores = np.nan_to_num(f1_scores, nan=0.0)
            if len(thresholds) > 0:
                best_idx = np.argmax(f1_scores[:-1])
                threshold = float(thresholds[best_idx])

        y_train_pred = (y_train_proba >= threshold).astype(int)
        y_test_pred = (y_test_proba >= threshold).astype(int)

        metrics = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "cv_best_auc": float(best_score),
            "train_auc": float(roc_auc_score(y_train, y_train_proba)),
            "test_auc": float(roc_auc_score(y_test, y_test_proba)),
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "train_bal_acc": float(balanced_accuracy_score(y_train, y_train_pred)),
            "test_bal_acc": float(balanced_accuracy_score(y_test, y_test_pred)),
            "train_precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
            "train_recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
            "train_f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
            "best_params": grid.best_params_,
            "n_nonzero_coefs": int(np.sum(self.model.coef_[0] != 0)),
            "resampling_strategy": best_strategy,
            "optimal_threshold": threshold,
        }

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        self._roc_data = {"fpr": fpr, "tpr": tpr, "auc": metrics["test_auc"]}
        self._test_data = (X_test, y_test, y_test_proba)

        return metrics

    def get_roc_data(self) -> dict[str, NDArray]:
        """Get ROC curve data after fitting with CV."""
        if not hasattr(self, "_roc_data"):
            raise ValueError("Must call fit_with_cv first")
        return self._roc_data


# =============================================================================
# Feature Selection Utilities
# =============================================================================


def select_top_features_by_correlation(
    X: pd.DataFrame,
    y: NDArray,
    feature_cols: list[str],
    k: int = 40,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Select top-k features by absolute Pearson correlation with target.

    This is useful for reducing dimensionality when you have many connectome
    features but want to focus on those most related to the outcome.

    Parameters
    ----------
    X : DataFrame
        Feature matrix with named columns.
    y : ndarray
        Target variable (binary or continuous).
    feature_cols : list of str
        Column names to consider for selection.
    k : int, default=40
        Number of top features to select.

    Returns
    -------
    selected : list of str
        Names of selected features.
    scores : list of tuple
        All (feature_name, abs_correlation) pairs, sorted descending.

    Examples
    --------
    >>> selected, scores = select_top_features_by_correlation(
    ...     X_train, y_train, connectome_cols, k=40
    ... )
    >>> X_train_selected = X_train[selected]
    """
    scores = []
    y_float = y.astype(float)

    for col in feature_cols:
        x = X[col].values.astype(float)
        mask = ~np.isnan(x)
        if mask.sum() < 10:
            scores.append((col, 0.0))
            continue
        x_mask = x[mask]
        y_mask = y_float[mask]
        if np.all(x_mask == x_mask[0]):
            scores.append((col, 0.0))
            continue
        corr = np.corrcoef(x_mask, y_mask)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        scores.append((col, abs(corr)))

    scores.sort(key=lambda t: t[1], reverse=True)
    selected = [name for name, _ in scores[:k]]
    return selected, scores


# =============================================================================
# Cognitive Feature Definitions (from HCP)
# =============================================================================

# Standard cognitive feature set used in HCP analyses
HCP_COGNITIVE_FEATURES = [
    # Fluid intelligence
    "PMAT24_A_CR",
    "PMAT24_A_SI",
    "PMAT24_A_RTCR",
    # Reading & vocabulary
    "ReadEng_Unadj",
    "ReadEng_AgeAdj",
    "PicVocab_Unadj",
    "PicVocab_AgeAdj",
    # Immediate & delayed word recall
    "IWRD_TOT",
    "IWRD_RTC",
    # Processing speed
    "ProcSpeed_Unadj",
    "ProcSpeed_AgeAdj",
    # Delay discounting
    "DDisc_SV_1mo_200",
    "DDisc_SV_6mo_200",
    "DDisc_SV_1yr_200",
    "DDisc_SV_3yr_200",
    "DDisc_SV_5yr_200",
    "DDisc_SV_10yr_200",
    "DDisc_SV_6mo_40K",
    "DDisc_SV_1yr_40K",
    "DDisc_SV_3yr_40K",
    "DDisc_SV_5yr_40K",
    "DDisc_SV_10yr_40K",
    "DDisc_AUC_200",
    "DDisc_AUC_40K",
    # Visuospatial / mental rotation
    "VSPLOT_TC",
    "VSPLOT_CRTE",
    "VSPLOT_OFF",
    # Sustained attention (SCPT)
    "SCPT_TP",
    "SCPT_TN",
    "SCPT_FP",
    "SCPT_FN",
    "SCPT_TPRT",
    "SCPT_SEN",
    "SCPT_SPEC",
    "SCPT_LRNR",
    # Working memory / list sorting
    "ListSort_Unadj",
    "ListSort_AgeAdj",
    # Episodic memory (picture sequence)
    "PicSeq_Unadj",
    "PicSeq_AgeAdj",
    # Socioeconomic covariates
    "SSAGA_Income",
    "SSAGA_Educ",
    # Executive function & attention
    "CardSort_Unadj",
    "CardSort_AgeAdj",
    "Flanker_Unadj",
    "Flanker_AgeAdj",
]


def get_cognitive_features(df: pd.DataFrame, include_age: bool = True) -> list[str]:
    """Get available cognitive features from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Data with potential cognitive feature columns.
    include_age : bool, default=True
        Whether to include age as a feature.

    Returns
    -------
    features : list of str
        List of cognitive feature column names present in df.
    """
    candidates = list(HCP_COGNITIVE_FEATURES)

    if include_age:
        if "Age_in_Yrs" in df.columns:
            candidates = ["Age_in_Yrs"] + candidates
        elif "Age" in df.columns:
            candidates = ["Age"] + candidates

    return [c for c in candidates if c in df.columns]


def get_connectome_features(df: pd.DataFrame, variant: str) -> list[str]:
    """Get connectome feature columns for a given representation.

    Parameters
    ----------
    df : DataFrame
        Data with connectome feature columns.
    variant : str
        One of 'tnpca', 'vae', 'pca' indicating the feature type.

    Returns
    -------
    features : list of str
        List of connectome feature column names.
    """
    if variant == "tnpca":
        return [c for c in df.columns if c.startswith("Struct_PC") or c.startswith("Func_PC")]
    elif variant == "vae":
        return [
            c for c in df.columns if c.startswith("VAE_Struct_LD") or c.startswith("VAE_Func_LD")
        ]
    elif variant == "pca":
        return [
            c for c in df.columns if c.startswith("Raw_Struct_PC") or c.startswith("Raw_Func_PC")
        ]
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'tnpca', 'vae', or 'pca'.")
