"""Tests for data preprocessing module."""

import numpy as np

from connectopy.data.preprocessing import (
    flatten_connectome,
    log_transform,
    minmax_normalize,
    preprocess_connectome,
    remove_zero_variance_features,
)


class TestFlattenConnectome:
    """Tests for flatten_connectome function."""

    def test_upper_triangle_shape(self, sample_connectome):
        """Test that upper triangle flattening produces correct shape."""
        flattened = flatten_connectome(sample_connectome, upper_triangle_only=True)

        n_regions = sample_connectome.shape[0]
        n_subjects = sample_connectome.shape[2]
        expected_features = n_regions * (n_regions - 1) // 2

        assert flattened.shape == (n_subjects, expected_features)

    def test_full_matrix_shape(self, sample_connectome):
        """Test that full matrix flattening produces correct shape."""
        flattened = flatten_connectome(sample_connectome, upper_triangle_only=False)

        n_regions = sample_connectome.shape[0]
        n_subjects = sample_connectome.shape[2]
        expected_features = n_regions * n_regions

        assert flattened.shape == (n_subjects, expected_features)

    def test_values_preserved(self, sample_connectome):
        """Test that values are correctly extracted."""
        flattened = flatten_connectome(sample_connectome, upper_triangle_only=True)

        # Check first subject's first value
        triu_indices = np.triu_indices(68, k=1)
        expected_first = sample_connectome[:, :, 0][triu_indices][0]

        assert flattened[0, 0] == expected_first


class TestRemoveZeroVarianceFeatures:
    """Tests for remove_zero_variance_features function."""

    def test_removes_constant_columns(self):
        """Test that constant columns are removed."""
        X = np.array(
            [
                [1, 2, 5],
                [2, 2, 6],
                [3, 2, 7],
            ]
        )

        X_filtered, mask = remove_zero_variance_features(X)

        assert X_filtered.shape == (3, 2)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_keeps_all_with_variance(self, sample_flattened_connectome):
        """Test that all columns are kept when all have variance."""
        X_filtered, mask = remove_zero_variance_features(sample_flattened_connectome)

        assert X_filtered.shape == sample_flattened_connectome.shape
        assert mask.all()


class TestLogTransform:
    """Tests for log_transform function."""

    def test_log_transform_positive(self):
        """Test log transform on positive values."""
        X = np.array([0, 1, 10, 100])
        X_log = log_transform(X)

        expected = np.log(X + 1)
        np.testing.assert_array_almost_equal(X_log, expected)

    def test_handles_zeros(self):
        """Test that zeros are handled correctly."""
        X = np.array([0, 0, 0])
        X_log = log_transform(X)

        assert np.all(np.isfinite(X_log))


class TestMinmaxNormalize:
    """Tests for minmax_normalize function."""

    def test_default_range(self):
        """Test normalization to [0, 1] range."""
        X = np.array([0, 50, 100])
        X_norm, _, _ = minmax_normalize(X)

        assert X_norm.min() == 0.0
        assert X_norm.max() == 1.0

    def test_custom_range(self):
        """Test normalization to custom range."""
        X = np.array([0, 50, 100])
        X_norm, _, _ = minmax_normalize(X, feature_range=(-1, 1))

        assert X_norm.min() == -1.0
        assert X_norm.max() == 1.0


class TestPreprocessConnectome:
    """Tests for preprocess_connectome function."""

    def test_full_pipeline(self, sample_connectome):
        """Test full preprocessing pipeline."""
        X_processed, metadata = preprocess_connectome(sample_connectome)

        n_subjects = sample_connectome.shape[2]

        assert X_processed.shape[0] == n_subjects
        assert "original_shape" in metadata
        assert metadata.get("log_transformed", False)
        assert metadata.get("normalized", False)

    def test_no_normalization(self, sample_connectome):
        """Test pipeline without normalization."""
        X_processed, metadata = preprocess_connectome(sample_connectome, normalize=False)

        assert "data_min" not in metadata
        assert "data_max" not in metadata
