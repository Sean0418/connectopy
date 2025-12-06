"""Principal Component Analysis for connectome data.

This module provides PCA functionality specifically designed for analyzing
brain connectome data, including variance explained calculations and
component interpretation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConnectomePCA:
    """PCA analysis for brain connectome data.

    This class wraps scikit-learn's PCA with additional functionality
    for connectome-specific analysis and reporting.

    Parameters
    ----------
    n_components : int, default=60
        Number of principal components to retain.
    scale : bool, default=True
        Whether to standardize features before PCA.

    Attributes
    ----------
    pca : PCA
        Fitted PCA model.
    scaler : StandardScaler or None
        Fitted scaler if scale=True.
    variance_explained_ : ndarray
        Variance explained by each component.
    cumulative_variance_ : ndarray
        Cumulative variance explained.

    Examples
    --------
    >>> pca_model = ConnectomePCA(n_components=60)
    >>> scores = pca_model.fit_transform(X)
    >>> print(f"Variance explained: {pca_model.total_variance_explained_:.2%}")
    """

    def __init__(self, n_components: int = 60, scale: bool = True) -> None:
        """Initialize the PCA model.

        Parameters
        ----------
        n_components : int, default=60
            Number of components to retain.
        scale : bool, default=True
            Whether to standardize features.
        """
        self.n_components = n_components
        self.scale = scale
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler() if scale else None

        # Attributes set after fitting
        self.variance_explained_: NDArray[np.float64] | None = None
        self.cumulative_variance_: NDArray[np.float64] | None = None
        self.total_variance_explained_: float | None = None

    def fit(self, X: NDArray[np.float64]) -> ConnectomePCA:
        """Fit the PCA model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : ConnectomePCA
            Fitted estimator.
        """
        X_scaled = self.scaler.fit_transform(X) if self.scaler is not None else X

        self.pca.fit(X_scaled)

        # Calculate variance statistics
        self.variance_explained_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.variance_explained_)
        self.total_variance_explained_ = self.cumulative_variance_[-1]

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform data to PCA space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_pca : ndarray of shape (n_samples, n_components)
            Transformed data in PCA space.
        """
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X

        return self.pca.transform(X_scaled)

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit the model and transform data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_pca : ndarray of shape (n_samples, n_components)
            Transformed data in PCA space.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform data back to original space.

        Parameters
        ----------
        X_pca : ndarray of shape (n_samples, n_components)
            Data in PCA space.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data in original space.
        """
        X_reconstructed = self.pca.inverse_transform(X_pca)

        if self.scaler is not None:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)

        return X_reconstructed

    def get_variance_report(self) -> pd.DataFrame:
        """Generate a report of variance explained by each component.

        Returns
        -------
        report : DataFrame
            DataFrame with columns for component index, individual variance
            explained, and cumulative variance explained.
        """
        if self.variance_explained_ is None:
            raise ValueError("Model must be fit before generating report")

        return pd.DataFrame(
            {
                "Component": range(1, self.n_components + 1),
                "Variance_Explained": self.variance_explained_,
                "Cumulative_Variance": self.cumulative_variance_,
            }
        )

    def to_dataframe(
        self,
        X: NDArray[np.float64],
        subject_ids: NDArray | None = None,
        prefix: str = "PC",
    ) -> pd.DataFrame:
        """Transform data and return as DataFrame with subject IDs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.
        subject_ids : ndarray, optional
            Subject identifiers. If None, uses integer indices.
        prefix : str, default="PC"
            Prefix for column names.

        Returns
        -------
        df : DataFrame
            DataFrame with Subject column and PC columns.
        """
        X_pca = self.transform(X)

        columns = [f"{prefix}{i + 1}" for i in range(self.n_components)]
        df = pd.DataFrame(X_pca, columns=columns)

        if subject_ids is not None:
            df.insert(0, "Subject", subject_ids)
        else:
            df.insert(0, "Subject", range(len(df)))

        return df
