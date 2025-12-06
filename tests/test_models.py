"""Tests for machine learning models."""

import numpy as np
import pandas as pd
import pytest

from brain_connectome.models.classifiers import ConnectomeRandomForest


class TestConnectomeRandomForest:
    """Tests for ConnectomeRandomForest class."""

    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        # Make class separable
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        feature_names = [f"Feature_{i}" for i in range(n_features)]

        return X, y, feature_names

    def test_fit(self, binary_classification_data):
        """Test model fitting."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeRandomForest(n_estimators=10, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        assert clf.feature_names is not None
        assert clf.feature_importances_ is not None

    def test_predict(self, binary_classification_data):
        """Test prediction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeRandomForest(n_estimators=10, random_state=42)
        clf.fit(X[:80], y[:80], feature_names=feature_names)

        predictions = clf.predict(X[80:])

        assert len(predictions) == 20
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, binary_classification_data):
        """Test probability prediction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeRandomForest(n_estimators=10, random_state=42)
        clf.fit(X[:80], y[:80], feature_names=feature_names)

        proba = clf.predict_proba(X[80:])

        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importance(self, binary_classification_data):
        """Test feature importance extraction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeRandomForest(n_estimators=10, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        importances = clf.feature_importances_

        assert isinstance(importances, pd.DataFrame)
        assert "Feature" in importances.columns
        assert "Importance" in importances.columns
        assert len(importances) == 20

    def test_get_top_features(self, binary_classification_data):
        """Test getting top features."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeRandomForest(n_estimators=10, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        top_features = clf.get_top_features(n=5)

        assert len(top_features) == 5
        # Should be sorted by importance descending
        assert top_features["Importance"].is_monotonic_decreasing

    def test_evaluate(self, binary_classification_data):
        """Test model evaluation."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeRandomForest(n_estimators=50, random_state=42)
        clf.fit(X[:80], y[:80], feature_names=feature_names)

        metrics = clf.evaluate(X[80:], y[80:])

        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_reproducibility(self, binary_classification_data):
        """Test that random_state ensures reproducibility."""
        X, y, feature_names = binary_classification_data

        clf1 = ConnectomeRandomForest(n_estimators=10, random_state=42)
        clf1.fit(X, y, feature_names=feature_names)
        pred1 = clf1.predict(X[:10])

        clf2 = ConnectomeRandomForest(n_estimators=10, random_state=42)
        clf2.fit(X, y, feature_names=feature_names)
        pred2 = clf2.predict(X[:10])

        np.testing.assert_array_equal(pred1, pred2)
