"""Classification models for brain connectome analysis.

This module provides wrapper classes for Random Forest and XGBoost classifiers
with additional functionality for feature importance analysis and
connectome-specific reporting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

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
                "interpret is required for ConnectomeEBM. " "Install with: pip install interpret"
            )

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
        for i, _ in enumerate(self.feature_names):
            if i < len(self.model.term_importances_):
                importances.append(self.model.term_importances_[i])
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
