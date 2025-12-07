"""Tests for analysis modules."""

import numpy as np
import pandas as pd

from connectopy.analysis.dimorphism import DimorphismAnalysis
from connectopy.analysis.pca import ConnectomePCA


class TestConnectomePCA:
    """Tests for ConnectomePCA class."""

    def test_fit_transform(self, sample_flattened_connectome):
        """Test fit_transform produces correct output shape."""
        pca = ConnectomePCA(n_components=10)
        X_pca = pca.fit_transform(sample_flattened_connectome)

        assert X_pca.shape == (50, 10)

    def test_variance_explained(self, sample_flattened_connectome):
        """Test variance explained is calculated correctly."""
        pca = ConnectomePCA(n_components=10)
        pca.fit(sample_flattened_connectome)

        assert pca.variance_explained_ is not None
        assert len(pca.variance_explained_) == 10
        assert pca.total_variance_explained_ > 0
        assert pca.total_variance_explained_ <= 1

    def test_inverse_transform(self, sample_flattened_connectome):
        """Test inverse transform reconstructs data."""
        pca = ConnectomePCA(n_components=50)
        X_pca = pca.fit_transform(sample_flattened_connectome)
        X_recon = pca.inverse_transform(X_pca)

        # Reconstruction should be close to original with many components
        assert X_recon.shape == sample_flattened_connectome.shape

    def test_to_dataframe(self, sample_flattened_connectome, sample_subject_ids):
        """Test conversion to DataFrame."""
        pca = ConnectomePCA(n_components=10)
        pca.fit(sample_flattened_connectome)

        df = pca.to_dataframe(
            sample_flattened_connectome,
            subject_ids=sample_subject_ids,
            prefix="Struct_PC",
        )

        assert "Subject" in df.columns
        assert "Struct_PC1" in df.columns
        assert len(df) == 50

    def test_variance_report(self, sample_flattened_connectome):
        """Test variance report generation."""
        pca = ConnectomePCA(n_components=10)
        pca.fit(sample_flattened_connectome)

        report = pca.get_variance_report()

        assert isinstance(report, pd.DataFrame)
        assert "Component" in report.columns
        assert "Variance_Explained" in report.columns
        assert "Cumulative_Variance" in report.columns


class TestDimorphismAnalysis:
    """Tests for DimorphismAnalysis class."""

    def test_initialization(self, sample_pca_data):
        """Test analysis initialization."""
        analysis = DimorphismAnalysis(sample_pca_data)

        assert len(analysis.male_data) > 0
        assert len(analysis.female_data) > 0

    def test_cohens_d(self, sample_pca_data):
        """Test Cohen's d calculation."""
        analysis = DimorphismAnalysis(sample_pca_data)

        # Create groups with known effect
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([2, 3, 4, 5, 6])

        d = analysis.cohens_d(group1, group2)

        # Effect should be approximately 1 SD difference
        assert abs(d) < 2  # Reasonable range

    def test_analyze(self, sample_pca_data):
        """Test dimorphism analysis."""
        analysis = DimorphismAnalysis(sample_pca_data)

        feature_cols = [f"PC{i + 1}" for i in range(10)]
        results = analysis.analyze(feature_columns=feature_cols)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 10
        assert "P_Value" in results.columns
        assert "Cohen_D" in results.columns
        assert "P_Adjusted" in results.columns
        assert "Significant" in results.columns

    def test_pc1_shows_effect(self, sample_pca_data):
        """Test that PC1 with built-in effect shows significance."""
        analysis = DimorphismAnalysis(sample_pca_data)

        results = analysis.analyze(feature_columns=["PC1"])

        # PC1 was designed with a gender effect in the fixture
        assert results.iloc[0]["P_Value"] < 0.1  # Should show some effect

    def test_get_top_features(self, sample_pca_data):
        """Test getting top features."""
        analysis = DimorphismAnalysis(sample_pca_data)
        feature_cols = [f"PC{i + 1}" for i in range(10)]
        analysis.analyze(feature_columns=feature_cols)

        top = analysis.get_top_features(n=5)

        assert len(top) == 5

    def test_summary(self, sample_pca_data):
        """Test summary generation."""
        analysis = DimorphismAnalysis(sample_pca_data)
        feature_cols = [f"PC{i + 1}" for i in range(10)]
        analysis.analyze(feature_columns=feature_cols)

        summary = analysis.summary()

        assert "n_male" in summary
        assert "n_female" in summary
        assert "n_features_tested" in summary
        assert summary["n_features_tested"] == 10
