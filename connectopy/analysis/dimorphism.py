"""Sexual dimorphism analysis for brain connectome data.

This module provides statistical tests and effect size calculations for
analyzing sex differences in brain connectivity patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DimorphismAnalysis:
    """Analyze sexual dimorphism in brain connectome features.

    This class provides methods for statistical testing of sex differences
    in brain connectivity, including t-tests, effect sizes, and multiple
    comparison correction.

    Parameters
    ----------
    data : DataFrame
        Dataset containing features and a gender column.
    gender_column : str, default="Gender"
        Name of the column containing gender labels.
    male_label : str, default="M"
        Label for male subjects.
    female_label : str, default="F"
        Label for female subjects.

    Attributes
    ----------
    data : DataFrame
        Input dataset.
    male_data : DataFrame
        Subset of data for male subjects.
    female_data : DataFrame
        Subset of data for female subjects.
    results : DataFrame or None
        Results of statistical tests after running analyze().

    Examples
    --------
    >>> analysis = DimorphismAnalysis(data)
    >>> results = analysis.analyze(feature_columns=["PC1", "PC2", "PC3"])
    >>> significant = results[results["Significant"]]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        gender_column: str = "Gender",
        male_label: str = "M",
        female_label: str = "F",
    ) -> None:
        """Initialize the analysis.

        Parameters
        ----------
        data : DataFrame
            Input dataset.
        gender_column : str
            Name of gender column.
        male_label : str
            Label for males.
        female_label : str
            Label for females.
        """
        self.data = data
        self.gender_column = gender_column
        self.male_label = male_label
        self.female_label = female_label

        # Split data by gender
        self.male_data = data[data[gender_column] == male_label]
        self.female_data = data[data[gender_column] == female_label]

        self.results: pd.DataFrame | None = None

    def cohens_d(
        self,
        group1: NDArray[np.float64],
        group2: NDArray[np.float64],
    ) -> float:
        """Calculate Cohen's d effect size.

        Parameters
        ----------
        group1 : ndarray
            First group values.
        group2 : ndarray
            Second group values.

        Returns
        -------
        d : float
            Cohen's d effect size.
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return float((np.mean(group1) - np.mean(group2)) / pooled_std)

    def analyze(
        self,
        feature_columns: list[str] | None = None,
        alpha: float = 0.05,
        correction_method: Literal["bonferroni", "fdr_bh", "none"] = "fdr_bh",
    ) -> pd.DataFrame:
        """Perform dimorphism analysis on specified features.

        Parameters
        ----------
        feature_columns : list of str, optional
            List of feature columns to analyze. If None, uses all numeric columns
            except the gender column.
        alpha : float, default=0.05
            Significance threshold.
        correction_method : {"bonferroni", "fdr_bh", "none"}, default="fdr_bh"
            Method for multiple comparison correction.

        Returns
        -------
        results : DataFrame
            Results with columns: Feature, T_Statistic, P_Value, Cohen_D,
            P_Adjusted, Significant.
        """
        if feature_columns is None:
            # Use all numeric columns except gender
            numeric_df: pd.DataFrame = self.data.select_dtypes(include=[np.number])
            feature_columns = [str(col) for col in numeric_df.columns if col != self.gender_column]

        results_list = []

        for feature in feature_columns:
            male_values = self.male_data[feature].dropna().values
            female_values = self.female_data[feature].dropna().values

            # T-test
            t_stat, p_value = stats.ttest_ind(male_values, female_values)

            # Effect size
            d = self.cohens_d(male_values, female_values)

            results_list.append(
                {
                    "Feature": feature,
                    "T_Statistic": t_stat,
                    "P_Value": p_value,
                    "Cohen_D": d,
                    "Male_Mean": np.mean(male_values),
                    "Female_Mean": np.mean(female_values),
                }
            )

        results = pd.DataFrame(results_list)

        # Multiple comparison correction
        p_values: NDArray[np.float64] = results["P_Value"].to_numpy()
        if correction_method == "bonferroni":
            p_adjusted = np.minimum(p_values * len(p_values), 1.0)
        elif correction_method == "fdr_bh":
            # Benjamini-Hochberg procedure
            n = len(p_values)
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            cummax_vals = np.maximum.accumulate(sorted_p * n / (np.arange(n) + 1)[::-1])[::-1]
            p_adjusted = np.empty_like(p_values)
            p_adjusted[sorted_idx] = np.minimum(cummax_vals, 1.0)
        else:
            p_adjusted = p_values

        results["P_Adjusted"] = p_adjusted
        results["Significant"] = results["P_Adjusted"] < alpha

        # Sort by effect size magnitude
        results = results.sort_values("Cohen_D", key=abs, ascending=False)
        results = results.reset_index(drop=True)

        self.results = results
        return results

    def get_top_features(
        self,
        n: int = 10,
        by: Literal["effect_size", "significance"] = "effect_size",
    ) -> pd.DataFrame:
        """Get top features showing dimorphism.

        Parameters
        ----------
        n : int, default=10
            Number of top features to return.
        by : {"effect_size", "significance"}, default="effect_size"
            Criterion for ranking features.

        Returns
        -------
        top_features : DataFrame
            Top n features.

        Raises
        ------
        ValueError
            If analyze() hasn't been called.
        """
        if self.results is None:
            raise ValueError("Must call analyze() first")

        if by == "effect_size":
            return self.results.head(n)
        else:
            return self.results.sort_values("P_Adjusted").head(n)

    def summary(self) -> dict:
        """Get summary statistics of the analysis.

        Returns
        -------
        summary : dict
            Dictionary with summary statistics.

        Raises
        ------
        ValueError
            If analyze() hasn't been called.
        """
        if self.results is None:
            raise ValueError("Must call analyze() first")

        n_significant = self.results["Significant"].sum()
        n_total = len(self.results)

        return {
            "n_male": len(self.male_data),
            "n_female": len(self.female_data),
            "n_features_tested": n_total,
            "n_significant": n_significant,
            "proportion_significant": n_significant / n_total,
            "largest_effect": self.results.iloc[0]["Cohen_D"],
            "most_significant_feature": self.results.sort_values("P_Adjusted").iloc[0]["Feature"],
        }
