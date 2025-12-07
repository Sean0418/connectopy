"""Tests for mediation analysis module."""

import numpy as np
import pandas as pd
import pytest

from connectopy.analysis.mediation import (
    MediationAnalysis,
    MediationResult,
    SexStratifiedMediation,
    SexStratifiedResult,
    run_multiple_mediations,
)


class TestMediationAnalysis:
    """Tests for MediationAnalysis class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known mediation structure."""
        rng = np.random.default_rng(42)
        n = 200

        # Create X (cognitive)
        X = rng.normal(0, 1, n)

        # Create M (brain network) - partially determined by X
        M = 0.5 * X + rng.normal(0, 0.5, n)

        # Create Y (alcohol) - determined by both X and M
        Y = 0.3 * X + 0.4 * M + rng.normal(0, 0.5, n)

        return X, M, Y

    def test_fit_returns_result(self, sample_data):
        """Test that fit returns a MediationResult."""
        X, M, Y = sample_data
        med = MediationAnalysis(n_bootstrap=100, random_state=42)
        result = med.fit(X, M, Y)

        assert isinstance(result, MediationResult)
        assert med.result_ is not None

    def test_path_coefficients(self, sample_data):
        """Test that path coefficients are computed correctly."""
        X, M, Y = sample_data
        med = MediationAnalysis(n_bootstrap=100, random_state=42)
        result = med.fit(X, M, Y)

        # Path a (X -> M) should be positive (~0.5)
        assert result.a > 0.3

        # Path b (M -> Y | X) should be positive (~0.4)
        assert result.b > 0.2

        # Indirect effect (a * b) should be positive
        assert result.indirect_effect > 0

        # Total effect (c) should be positive
        assert result.c > 0

    def test_confidence_intervals(self, sample_data):
        """Test that confidence intervals are computed."""
        X, M, Y = sample_data
        med = MediationAnalysis(n_bootstrap=500, random_state=42)
        result = med.fit(X, M, Y)

        # CI should bracket the indirect effect
        assert result.ci_low < result.indirect_effect < result.ci_high

        # With our data, effect should be significant
        assert result.significant

    def test_sobel_test(self, sample_data):
        """Test Sobel test computation."""
        X, M, Y = sample_data
        med = MediationAnalysis(n_bootstrap=100, random_state=42)
        result = med.fit(X, M, Y)

        # Sobel p-value should be valid
        assert 0 <= result.sobel_p <= 1

        # With strong mediation, should be significant
        assert result.sobel_p < 0.05

    def test_result_to_dict(self, sample_data):
        """Test conversion to dictionary."""
        X, M, Y = sample_data
        med = MediationAnalysis(n_bootstrap=100, random_state=42)
        result = med.fit(X, M, Y)

        d = result.to_dict()
        assert "a" in d
        assert "b" in d
        assert "indirect_effect" in d
        assert "significant" in d

    def test_handles_nan(self):
        """Test that NaN values are handled."""
        rng = np.random.default_rng(42)
        n = 100

        X = rng.normal(0, 1, n)
        M = 0.5 * X + rng.normal(0, 0.5, n)
        Y = 0.3 * X + 0.4 * M + rng.normal(0, 0.5, n)

        # Add some NaN values
        X[0] = np.nan
        M[5] = np.nan
        Y[10] = np.nan

        med = MediationAnalysis(n_bootstrap=100, random_state=42)
        result = med.fit(X, M, Y)

        # Should still compute valid results
        assert np.isfinite(result.indirect_effect)


class TestSexStratifiedMediation:
    """Tests for SexStratifiedMediation class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame with sex-specific mediation."""
        rng = np.random.default_rng(42)
        n = 300

        gender = rng.choice(["M", "F"], n)

        # Cognitive
        X = rng.normal(0, 1, n)

        # Brain - different mediation strength by sex
        M = np.where(
            gender == "M",
            0.6 * X + rng.normal(0, 0.5, n),  # Strong for males
            0.2 * X + rng.normal(0, 0.5, n),  # Weak for females
        )

        # Alcohol
        Y = 0.3 * X + 0.4 * M + rng.normal(0, 0.5, n)

        return pd.DataFrame(
            {
                "Gender": gender,
                "Cognitive": X,
                "Brain": M,
                "Alcohol": Y,
            }
        )

    def test_fit_returns_result(self, sample_dataframe):
        """Test that fit returns a SexStratifiedResult."""
        ssm = SexStratifiedMediation(n_bootstrap=100, random_state=42)
        result = ssm.fit(
            sample_dataframe,
            cognitive_col="Cognitive",
            brain_col="Brain",
            alcohol_col="Alcohol",
            sex_col="Gender",
        )

        assert isinstance(result, SexStratifiedResult)
        assert isinstance(result.male, MediationResult)
        assert isinstance(result.female, MediationResult)

    def test_sex_difference_detected(self, sample_dataframe):
        """Test that sex differences are detected when present."""
        ssm = SexStratifiedMediation(n_bootstrap=500, random_state=42)
        result = ssm.fit(
            sample_dataframe,
            cognitive_col="Cognitive",
            brain_col="Brain",
            alcohol_col="Alcohol",
        )

        # Male indirect effect should be larger
        assert result.male.indirect_effect > result.female.indirect_effect

        # Difference should be computed
        expected_diff = result.male.indirect_effect - result.female.indirect_effect
        assert abs(result.difference - expected_diff) < 0.001

    def test_result_to_dict(self, sample_dataframe):
        """Test conversion to dictionary."""
        ssm = SexStratifiedMediation(n_bootstrap=100, random_state=42)
        result = ssm.fit(
            sample_dataframe,
            cognitive_col="Cognitive",
            brain_col="Brain",
            alcohol_col="Alcohol",
        )

        d = result.to_dict()
        assert "male" in d
        assert "female" in d
        assert "difference" in d
        assert "diff_significant" in d

    def test_insufficient_samples_error(self):
        """Test error when insufficient samples in a group."""
        df = pd.DataFrame(
            {
                "Gender": ["M"] * 20 + ["F"] * 5,  # Too few females
                "Cognitive": np.random.randn(25),
                "Brain": np.random.randn(25),
                "Alcohol": np.random.randn(25),
            }
        )

        ssm = SexStratifiedMediation(n_bootstrap=100)

        with pytest.raises(ValueError, match="Insufficient samples"):
            ssm.fit(df, "Cognitive", "Brain", "Alcohol")


class TestRunMultipleMediations:
    """Tests for run_multiple_mediations function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame with multiple variables."""
        rng = np.random.default_rng(42)
        n = 200

        return pd.DataFrame(
            {
                "Gender": rng.choice(["M", "F"], n),
                "Cog1": rng.normal(0, 1, n),
                "Cog2": rng.normal(0, 1, n),
                "Brain1": rng.normal(0, 1, n),
                "Brain2": rng.normal(0, 1, n),
                "Alcohol": rng.normal(0, 1, n),
            }
        )

    def test_returns_dataframe(self, sample_dataframe):
        """Test that function returns a DataFrame."""
        results = run_multiple_mediations(
            sample_dataframe,
            cognitive_cols=["Cog1", "Cog2"],
            brain_cols=["Brain1"],
            alcohol_col="Alcohol",
            n_bootstrap=50,
            random_state=42,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # 2 cognitive × 1 brain

    def test_correct_columns(self, sample_dataframe):
        """Test that result has expected columns."""
        results = run_multiple_mediations(
            sample_dataframe,
            cognitive_cols=["Cog1"],
            brain_cols=["Brain1"],
            alcohol_col="Alcohol",
            n_bootstrap=50,
            random_state=42,
        )

        expected_cols = [
            "cognitive",
            "brain_network",
            "male_indirect",
            "female_indirect",
            "sex_difference",
            "sex_diff_significant",
        ]

        for col in expected_cols:
            assert col in results.columns
