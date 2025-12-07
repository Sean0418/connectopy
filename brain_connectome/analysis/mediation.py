"""Mediation analysis for brain-behavior relationships.

This module implements mediation analysis to test whether brain networks
mediate the relationship between cognitive traits and alcohol dependence,
with support for sex-stratified comparisons.

Model: X (cognitive) → M (brain network) → Y (alcohol)

References
----------
    Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable
    distinction in social psychological research.
    Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling
    strategies for assessing and comparing indirect effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MediationResult:
    """Results from a mediation analysis.

    Attributes
    ----------
        a: Path coefficient from X to M (cognitive → brain).
        b: Path coefficient from M to Y controlling for X (brain → alcohol).
        c: Total effect of X on Y (cognitive → alcohol).
        c_prime: Direct effect of X on Y controlling for M.
        indirect_effect: Product of a and b (mediation effect).
        ci_low: Lower bound of bootstrap confidence interval for indirect effect.
        ci_high: Upper bound of bootstrap confidence interval for indirect effect.
        sobel_z: Sobel test z-statistic.
        sobel_p: Sobel test p-value.
        proportion_mediated: Proportion of total effect mediated (ab/c).
        significant: Whether the indirect effect is significant (CI excludes 0).
    """

    a: float
    b: float
    c: float
    c_prime: float
    indirect_effect: float
    ci_low: float
    ci_high: float
    sobel_z: float
    sobel_p: float
    proportion_mediated: float
    significant: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "c_prime": self.c_prime,
            "indirect_effect": self.indirect_effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "sobel_z": self.sobel_z,
            "sobel_p": self.sobel_p,
            "proportion_mediated": self.proportion_mediated,
            "significant": self.significant,
        }


class MediationAnalysis:
    """Test if brain networks mediate cognitive-alcohol relationships.

    This class implements the Baron & Kenny (1986) mediation framework with
    bootstrap confidence intervals for the indirect effect.

    Model:
        X (cognitive) → M (brain network) → Y (alcohol)

    Paths:
        a: X → M (cognitive affects brain)
        b: M → Y (brain affects alcohol, controlling for X)
        c: X → Y (total effect)
        c': X → Y controlling for M (direct effect)
        ab: indirect effect (mediation)

    Example:
        >>> med = MediationAnalysis(n_bootstrap=5000)
        >>> result = med.fit(cognitive_scores, brain_pcs, alcohol_scores)
        >>> print(f"Indirect effect: {result.indirect_effect:.4f}")
        >>> print(f"Significant: {result.significant}")
    """

    def __init__(
        self,
        n_bootstrap: int = 5000,
        confidence: float = 0.95,
        random_state: int | None = None,
    ) -> None:
        """Initialize the mediation analysis.

        Args:
            n_bootstrap: Number of bootstrap samples for confidence intervals.
            confidence: Confidence level for intervals (default 0.95 for 95% CI).
            random_state: Random seed for reproducibility.
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.random_state = random_state
        self.result_: MediationResult | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        M: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> MediationResult:
        """Run mediation analysis with bootstrap confidence intervals.

        Args:
            X: Predictor variable (cognitive trait), shape (n_samples,) or (n, 1).
            M: Mediator variable (brain network feature), shape (n_samples,) or (n, 1).
            Y: Outcome variable (alcohol dependence), shape (n_samples,).

        Returns
        -------
            MediationResult containing all path coefficients and test statistics.
        """
        # Ensure proper shapes
        X = np.asarray(X).ravel()
        M = np.asarray(M).ravel()
        Y = np.asarray(Y).ravel()

        if len(X) != len(M) or len(X) != len(Y):
            msg = "X, M, and Y must have the same length"
            raise ValueError(msg)

        # Remove any NaN values
        mask = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
        X, M, Y = X[mask], M[mask], Y[mask]

        # Path a: X → M
        X_2d = X.reshape(-1, 1)
        model_a = LinearRegression().fit(X_2d, M)
        a = float(model_a.coef_[0])

        # Path b and c': [X, M] → Y
        XM = np.column_stack([X, M])
        model_b = LinearRegression().fit(XM, Y)
        c_prime = float(model_b.coef_[0])  # Direct effect
        b = float(model_b.coef_[1])  # b path

        # Path c: X → Y (total effect)
        model_c = LinearRegression().fit(X_2d, Y)
        c = float(model_c.coef_[0])

        # Indirect effect
        ab = a * b

        # Bootstrap for confidence intervals
        rng = np.random.default_rng(self.random_state)
        ab_boots = self._bootstrap_indirect(X, M, Y, rng)
        alpha = 1 - self.confidence
        ci_low = float(np.percentile(ab_boots, alpha / 2 * 100))
        ci_high = float(np.percentile(ab_boots, (1 - alpha / 2) * 100))

        # Sobel test
        se_a = self._standard_error(X_2d, M, model_a)
        se_b = self._standard_error(XM, Y, model_b, coef_idx=1)
        sobel_se = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
        sobel_z = ab / sobel_se if sobel_se > 0 else 0.0
        sobel_p = float(2 * (1 - stats.norm.cdf(abs(sobel_z))))

        # Proportion mediated
        prop_mediated = ab / c if abs(c) > 1e-10 else np.nan

        # Significance: CI doesn't include zero
        significant = (ci_low > 0) or (ci_high < 0)

        self.result_ = MediationResult(
            a=a,
            b=b,
            c=c,
            c_prime=c_prime,
            indirect_effect=ab,
            ci_low=ci_low,
            ci_high=ci_high,
            sobel_z=sobel_z,
            sobel_p=sobel_p,
            proportion_mediated=prop_mediated,
            significant=significant,
        )

        return self.result_

    def _bootstrap_indirect(
        self,
        X: NDArray[np.float64],
        M: NDArray[np.float64],
        Y: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Bootstrap the indirect effect."""
        n = len(X)
        ab_boots = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            X_b, M_b, Y_b = X[idx], M[idx], Y[idx]

            # Fit models on bootstrap sample
            X_b_2d = X_b.reshape(-1, 1)
            a_boot = LinearRegression().fit(X_b_2d, M_b).coef_[0]

            XM_b = np.column_stack([X_b, M_b])
            b_boot = LinearRegression().fit(XM_b, Y_b).coef_[1]

            ab_boots[i] = a_boot * b_boot

        return ab_boots

    def _standard_error(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        model: LinearRegression,
        coef_idx: int = 0,
    ) -> float:
        """Calculate standard error of a regression coefficient."""
        n = len(y)
        p = X.shape[1] if X.ndim > 1 else 1

        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = float(np.sum(residuals**2) / (n - p - 1))

        # Variance-covariance matrix
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        var_coef = mse * XtX_inv

        return float(np.sqrt(var_coef[coef_idx, coef_idx]))


@dataclass
class SexStratifiedResult:
    """Results from sex-stratified mediation analysis.

    Attributes
    ----------
        male: Mediation result for males.
        female: Mediation result for females.
        difference: Difference in indirect effects (male - female).
        diff_ci_low: Lower CI bound for difference.
        diff_ci_high: Upper CI bound for difference.
        diff_significant: Whether sex difference is significant.
    """

    male: MediationResult
    female: MediationResult
    difference: float
    diff_ci_low: float
    diff_ci_high: float
    diff_significant: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "male": self.male.to_dict(),
            "female": self.female.to_dict(),
            "difference": self.difference,
            "diff_ci_low": self.diff_ci_low,
            "diff_ci_high": self.diff_ci_high,
            "diff_significant": self.diff_significant,
        }


class SexStratifiedMediation:
    """Compare mediation effects across sexes.

    This class runs separate mediation analyses for males and females,
    then tests whether the indirect effects differ significantly between groups.

    Example:
        >>> ssm = SexStratifiedMediation(n_bootstrap=1000)
        >>> result = ssm.fit(
        ...     data=df,
        ...     cognitive_col="FluidIntelligence",
        ...     brain_col="SC_PC1",
        ...     alcohol_col="AlcoholDependence",
        ...     sex_col="Gender",
        ... )
        >>> print(f"Male indirect: {result.male.indirect_effect:.4f}")
        >>> print(f"Female indirect: {result.female.indirect_effect:.4f}")
        >>> print(f"Sex difference significant: {result.diff_significant}")
    """

    def __init__(
        self,
        n_bootstrap: int = 5000,
        confidence: float = 0.95,
        random_state: int | None = None,
    ) -> None:
        """Initialize sex-stratified mediation analysis.

        Args:
            n_bootstrap: Number of bootstrap samples.
            confidence: Confidence level for intervals.
            random_state: Random seed for reproducibility.
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.random_state = random_state
        self.result_: SexStratifiedResult | None = None

    def fit(
        self,
        data: pd.DataFrame,
        cognitive_col: str,
        brain_col: str,
        alcohol_col: str,
        sex_col: str = "Gender",
        male_value: str = "M",
    ) -> SexStratifiedResult:
        """Run mediation analysis separately by sex and test difference.

        Args:
            data: DataFrame containing all variables.
            cognitive_col: Column name for cognitive predictor.
            brain_col: Column name for brain network mediator.
            alcohol_col: Column name for alcohol outcome.
            sex_col: Column name for sex variable.
            male_value: Value indicating male subjects.

        Returns
        -------
            SexStratifiedResult with results for each sex and difference test.
        """
        # Split by sex
        males = data[data[sex_col] == male_value].copy()
        females = data[data[sex_col] != male_value].copy()

        if len(males) < 30 or len(females) < 30:
            msg = f"Insufficient samples: {len(males)} males, {len(females)} females"
            raise ValueError(msg)

        # Run mediation for each sex
        rng = np.random.default_rng(self.random_state)

        med_male = MediationAnalysis(
            n_bootstrap=self.n_bootstrap,
            confidence=self.confidence,
            random_state=int(rng.integers(0, 2**31)),
        )
        male_result = med_male.fit(
            males[cognitive_col].values,
            males[brain_col].values,
            males[alcohol_col].values,
        )

        med_female = MediationAnalysis(
            n_bootstrap=self.n_bootstrap,
            confidence=self.confidence,
            random_state=int(rng.integers(0, 2**31)),
        )
        female_result = med_female.fit(
            females[cognitive_col].values,
            females[brain_col].values,
            females[alcohol_col].values,
        )

        # Test difference in indirect effects via bootstrap
        diff = male_result.indirect_effect - female_result.indirect_effect
        diff_boots = self._bootstrap_difference(
            data,
            cognitive_col,
            brain_col,
            alcohol_col,
            sex_col,
            male_value,
            rng,
        )

        alpha = 1 - self.confidence
        diff_ci_low = float(np.percentile(diff_boots, alpha / 2 * 100))
        diff_ci_high = float(np.percentile(diff_boots, (1 - alpha / 2) * 100))
        diff_significant = (diff_ci_low > 0) or (diff_ci_high < 0)

        self.result_ = SexStratifiedResult(
            male=male_result,
            female=female_result,
            difference=diff,
            diff_ci_low=diff_ci_low,
            diff_ci_high=diff_ci_high,
            diff_significant=diff_significant,
        )

        return self.result_

    def _bootstrap_difference(
        self,
        data: pd.DataFrame,
        cog_col: str,
        brain_col: str,
        alc_col: str,
        sex_col: str,
        male_val: str,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Bootstrap the difference in indirect effects between sexes."""
        # Use fewer bootstrap samples for the difference test
        n_boot = min(1000, self.n_bootstrap)
        diffs = []

        for _ in range(n_boot):
            # Resample entire dataset
            boot_data = data.sample(n=len(data), replace=True, random_state=rng)

            males = boot_data[boot_data[sex_col] == male_val]
            females = boot_data[boot_data[sex_col] != male_val]

            # Skip if insufficient samples in either group
            if len(males) < 20 or len(females) < 20:
                continue

            try:
                # Quick mediation (fewer bootstrap samples for inner loop)
                med_m = MediationAnalysis(n_bootstrap=100, random_state=int(rng.integers(0, 2**31)))
                med_f = MediationAnalysis(n_bootstrap=100, random_state=int(rng.integers(0, 2**31)))

                res_m = med_m.fit(
                    males[cog_col].values,
                    males[brain_col].values,
                    males[alc_col].values,
                )
                res_f = med_f.fit(
                    females[cog_col].values,
                    females[brain_col].values,
                    females[alc_col].values,
                )

                diffs.append(res_m.indirect_effect - res_f.indirect_effect)
            except (ValueError, np.linalg.LinAlgError):
                # Skip failed iterations
                continue

        return np.array(diffs) if diffs else np.array([0.0])


def run_multiple_mediations(
    data: pd.DataFrame,
    cognitive_cols: list[str],
    brain_cols: list[str],
    alcohol_col: str,
    sex_col: str = "Gender",
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Run mediation analysis for multiple cognitive-brain combinations.

    This is a convenience function that tests all combinations of cognitive
    predictors and brain network mediators, stratified by sex.

    Args:
        data: DataFrame with all variables.
        cognitive_cols: List of cognitive predictor column names.
        brain_cols: List of brain network mediator column names.
        alcohol_col: Alcohol outcome column name.
        sex_col: Sex variable column name.
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed.

    Returns
    -------
        DataFrame with results for all combinations.
    """
    results = []
    rng = np.random.default_rng(random_state)

    total = len(cognitive_cols) * len(brain_cols)
    current = 0

    for cog_var in cognitive_cols:
        for brain_var in brain_cols:
            current += 1
            print(f"[{current}/{total}] Testing: {cog_var} → {brain_var} → {alcohol_col}")

            try:
                ssm = SexStratifiedMediation(
                    n_bootstrap=n_bootstrap,
                    random_state=int(rng.integers(0, 2**31)),
                )
                result = ssm.fit(
                    data=data,
                    cognitive_col=cog_var,
                    brain_col=brain_var,
                    alcohol_col=alcohol_col,
                    sex_col=sex_col,
                )

                results.append(
                    {
                        "cognitive": cog_var,
                        "brain_network": brain_var,
                        "male_indirect": result.male.indirect_effect,
                        "male_ci_low": result.male.ci_low,
                        "male_ci_high": result.male.ci_high,
                        "male_significant": result.male.significant,
                        "female_indirect": result.female.indirect_effect,
                        "female_ci_low": result.female.ci_low,
                        "female_ci_high": result.female.ci_high,
                        "female_significant": result.female.significant,
                        "sex_difference": result.difference,
                        "sex_diff_ci_low": result.diff_ci_low,
                        "sex_diff_ci_high": result.diff_ci_high,
                        "sex_diff_significant": result.diff_significant,
                    }
                )
            except Exception as e:
                print(f"  Skipped due to error: {e}")
                continue

    return pd.DataFrame(results)
