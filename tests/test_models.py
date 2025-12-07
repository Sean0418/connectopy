"""Tests for machine learning models."""

import numpy as np
import pandas as pd
import pytest

from connectopy.models.classifiers import (
    ConnectomeLogistic,
    ConnectomeRandomForest,
    ConnectomeSVM,
    get_cognitive_features,
    get_connectome_features,
    select_top_features_by_correlation,
)

# Check if interpret package is installed (required for EBM)
try:
    from interpret.glassbox import ExplainableBoostingClassifier  # noqa: F401

    from connectopy.models.classifiers import ConnectomeEBM

    EBM_AVAILABLE = True
except ImportError:
    EBM_AVAILABLE = False
    ConnectomeEBM = None  # type: ignore[misc, assignment]


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


@pytest.mark.skipif(not EBM_AVAILABLE, reason="interpret package not installed")
class TestConnectomeEBM:
    """Tests for ConnectomeEBM (Explainable Boosting Machine) class."""

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
        """Test EBM fitting."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeEBM(max_bins=32, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        assert clf.feature_names is not None
        assert clf.feature_importances_ is not None

    def test_predict(self, binary_classification_data):
        """Test EBM prediction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeEBM(max_bins=32, random_state=42)
        clf.fit(X[:80], y[:80], feature_names=feature_names)

        predictions = clf.predict(X[80:])

        assert len(predictions) == 20
        assert set(predictions).issubset({0, 1})

    def test_feature_importance(self, binary_classification_data):
        """Test EBM feature importance extraction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeEBM(max_bins=32, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        importances = clf.feature_importances_

        assert isinstance(importances, pd.DataFrame)
        assert "Feature" in importances.columns
        assert "Importance" in importances.columns

    def test_evaluate(self, binary_classification_data):
        """Test EBM evaluation."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeEBM(max_bins=32, random_state=42)
        clf.fit(X[:80], y[:80], feature_names=feature_names)

        metrics = clf.evaluate(X[80:], y[80:])

        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_explain_global(self, binary_classification_data):
        """Test EBM global explanation."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeEBM(max_bins=32, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        explanation = clf.explain_global()

        # EBM returns an explanation object
        assert explanation is not None

    def test_explain_local(self, binary_classification_data):
        """Test EBM local explanation."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeEBM(max_bins=32, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        explanation = clf.explain_local(X[:5])

        # EBM returns an explanation object
        assert explanation is not None


class TestConnectomeSVM:
    """Tests for ConnectomeSVM class."""

    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        feature_names = [f"Feature_{i}" for i in range(n_features)]

        return X, y, feature_names

    def test_fit(self, binary_classification_data):
        """Test SVM fitting."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeSVM(C=1.0, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        assert clf.feature_names is not None
        assert clf._scaler is not None

    def test_predict(self, binary_classification_data):
        """Test SVM prediction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeSVM(C=1.0, random_state=42)
        clf.fit(X[:80], y[:80], feature_names=feature_names)

        predictions = clf.predict(X[80:])
        assert len(predictions) == 20
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, binary_classification_data):
        """Test SVM probability prediction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeSVM(C=1.0, random_state=42)
        clf.fit(X[:80], y[:80], feature_names=feature_names)

        proba = clf.predict_proba(X[80:])
        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestConnectomeLogistic:
    """Tests for ConnectomeLogistic class."""

    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        feature_names = [f"Feature_{i}" for i in range(n_features)]

        return X, y, feature_names

    def test_fit(self, binary_classification_data):
        """Test Logistic fitting."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeLogistic(C=1.0, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        assert clf.feature_names is not None
        assert clf.feature_importances_ is not None

    def test_predict(self, binary_classification_data):
        """Test Logistic prediction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeLogistic(C=1.0, random_state=42)
        clf.fit(X[:80], y[:80], feature_names=feature_names)

        predictions = clf.predict(X[80:])
        assert len(predictions) == 20
        assert set(predictions).issubset({0, 1})

    def test_get_coefficients(self, binary_classification_data):
        """Test Logistic coefficient extraction."""
        X, y, feature_names = binary_classification_data

        clf = ConnectomeLogistic(C=1.0, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        coefs = clf.get_coefficients()
        assert "Feature" in coefs.columns
        assert "Coefficient" in coefs.columns
        assert "AbsCoef" in coefs.columns
        assert len(coefs) == 20


class TestFitWithCV:
    """Tests for fit_with_cv method on classifiers."""

    @pytest.fixture
    def imbalanced_classification_data(self):
        """Generate imbalanced binary classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 15

        X = np.random.randn(n_samples, n_features)
        # Create imbalanced classes (20% positive)
        y = np.zeros(n_samples, dtype=int)
        y[:40] = 1
        np.random.shuffle(y)

        feature_names = [f"Feature_{i}" for i in range(n_features)]

        return X, y, feature_names

    def test_rf_fit_with_cv(self, imbalanced_classification_data):
        """Test Random Forest fit_with_cv method."""
        X, y, feature_names = imbalanced_classification_data

        clf = ConnectomeRandomForest(n_estimators=50, random_state=42)
        metrics = clf.fit_with_cv(
            X,
            y,
            feature_names=feature_names,
            handle_imbalance=True,
            param_grid={"rf__n_estimators": [50], "rf__max_depth": [5]},
        )

        # Check metrics are returned
        assert "cv_best_auc" in metrics
        assert "test_auc" in metrics
        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics
        assert "best_params" in metrics

        # Check ROC data is stored
        roc_data = clf.get_roc_data()
        assert "fpr" in roc_data
        assert "tpr" in roc_data
        assert "auc" in roc_data

    @pytest.mark.skipif(not EBM_AVAILABLE, reason="interpret package not installed")
    def test_ebm_fit_with_cv(self, imbalanced_classification_data):
        """Test EBM fit_with_cv method."""
        X, y, feature_names = imbalanced_classification_data

        clf = ConnectomeEBM(max_bins=16, random_state=42)
        metrics = clf.fit_with_cv(
            X,
            y,
            feature_names=feature_names,
            handle_imbalance=True,
            param_grid={"max_leaves": [2], "min_samples_leaf": [20]},
        )

        # Check metrics are returned
        assert "cv_best_auc" in metrics
        assert "test_auc" in metrics
        assert "n_features_used" in metrics

        # Check ROC data is stored
        roc_data = clf.get_roc_data()
        assert "fpr" in roc_data
        assert "tpr" in roc_data

    def test_svm_fit_with_cv(self, imbalanced_classification_data):
        """Test SVM fit_with_cv method."""
        X, y, feature_names = imbalanced_classification_data

        clf = ConnectomeSVM(random_state=42)
        metrics = clf.fit_with_cv(
            X,
            y,
            feature_names=feature_names,
            param_grid={"svm__C": [1], "svm__kernel": ["rbf"]},
            select_k_best=10,
            optimize_threshold=True,
        )

        # Check metrics are returned
        assert "cv_best_auc" in metrics
        assert "test_auc" in metrics
        assert "optimal_threshold" in metrics
        assert "resampling_strategy" in metrics

        # Check ROC data is stored
        roc_data = clf.get_roc_data()
        assert "fpr" in roc_data
        assert "tpr" in roc_data

    def test_logistic_fit_with_cv(self, imbalanced_classification_data):
        """Test Logistic fit_with_cv method."""
        X, y, feature_names = imbalanced_classification_data

        clf = ConnectomeLogistic(random_state=42)
        metrics = clf.fit_with_cv(
            X,
            y,
            feature_names=feature_names,
            param_grid=[
                {"logistic__C": [1], "logistic__penalty": ["l2"], "logistic__solver": ["lbfgs"]}
            ],
            select_k_best=10,
            optimize_threshold=True,
        )

        # Check metrics are returned
        assert "cv_best_auc" in metrics
        assert "test_auc" in metrics
        assert "n_nonzero_coefs" in metrics
        assert "optimal_threshold" in metrics

        # Check ROC data is stored
        roc_data = clf.get_roc_data()
        assert "fpr" in roc_data
        assert "tpr" in roc_data


class TestFeatureUtilities:
    """Tests for feature selection and extraction utilities."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame with various feature types."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "Subject": range(n_samples),
            "Gender": np.random.choice(["M", "F"], n_samples),
            "Age_in_Yrs": np.random.randint(22, 36, n_samples),
            # Cognitive features
            "PMAT24_A_CR": np.random.randn(n_samples),
            "ReadEng_Unadj": np.random.randn(n_samples),
            "ListSort_Unadj": np.random.randn(n_samples),
            # TNPCA features
            "Struct_PC1": np.random.randn(n_samples),
            "Struct_PC2": np.random.randn(n_samples),
            "Func_PC1": np.random.randn(n_samples),
            # VAE features
            "VAE_Struct_LD1": np.random.randn(n_samples),
            "VAE_Func_LD1": np.random.randn(n_samples),
            # Raw PCA features
            "Raw_Struct_PC1": np.random.randn(n_samples),
            "Raw_Func_PC1": np.random.randn(n_samples),
        }
        return pd.DataFrame(data)

    def test_get_cognitive_features(self, sample_dataframe):
        """Test get_cognitive_features function."""
        cog_features = get_cognitive_features(sample_dataframe, include_age=True)

        assert "Age_in_Yrs" in cog_features
        assert "PMAT24_A_CR" in cog_features
        assert "ReadEng_Unadj" in cog_features
        assert "ListSort_Unadj" in cog_features
        # Should not include non-cognitive features
        assert "Struct_PC1" not in cog_features
        assert "Subject" not in cog_features

    def test_get_cognitive_features_no_age(self, sample_dataframe):
        """Test get_cognitive_features without age."""
        cog_features = get_cognitive_features(sample_dataframe, include_age=False)

        assert "Age_in_Yrs" not in cog_features
        assert "PMAT24_A_CR" in cog_features

    def test_get_connectome_features_tnpca(self, sample_dataframe):
        """Test get_connectome_features for TNPCA variant."""
        conn_features = get_connectome_features(sample_dataframe, "tnpca")

        assert "Struct_PC1" in conn_features
        assert "Struct_PC2" in conn_features
        assert "Func_PC1" in conn_features
        # Should not include other variants
        assert "VAE_Struct_LD1" not in conn_features
        assert "Raw_Struct_PC1" not in conn_features

    def test_get_connectome_features_vae(self, sample_dataframe):
        """Test get_connectome_features for VAE variant."""
        conn_features = get_connectome_features(sample_dataframe, "vae")

        assert "VAE_Struct_LD1" in conn_features
        assert "VAE_Func_LD1" in conn_features
        assert "Struct_PC1" not in conn_features

    def test_get_connectome_features_pca(self, sample_dataframe):
        """Test get_connectome_features for raw PCA variant."""
        conn_features = get_connectome_features(sample_dataframe, "pca")

        assert "Raw_Struct_PC1" in conn_features
        assert "Raw_Func_PC1" in conn_features
        assert "Struct_PC1" not in conn_features

    def test_get_connectome_features_invalid(self, sample_dataframe):
        """Test get_connectome_features with invalid variant."""
        with pytest.raises(ValueError, match="Unknown variant"):
            get_connectome_features(sample_dataframe, "invalid_variant")

    def test_select_top_features_by_correlation(self, sample_dataframe):
        """Test select_top_features_by_correlation function."""
        # Create a target correlated with Struct_PC1
        y = sample_dataframe["Struct_PC1"].values + np.random.randn(100) * 0.1
        y = (y > y.mean()).astype(int)

        feature_cols = ["Struct_PC1", "Struct_PC2", "Func_PC1", "PMAT24_A_CR"]

        selected, scores = select_top_features_by_correlation(
            sample_dataframe, y, feature_cols, k=2
        )

        assert len(selected) == 2
        assert len(scores) == 4
        # Struct_PC1 should be most correlated (since y was derived from it)
        assert selected[0] == "Struct_PC1"
        # Scores should be sorted descending by abs correlation
        assert scores[0][1] >= scores[1][1]
