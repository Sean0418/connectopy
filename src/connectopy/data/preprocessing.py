"""Preprocessing utilities for connectome data.

This module provides functions for preprocessing raw connectome matrices,
including flattening, normalization, and handling missing data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def flatten_connectome(
    connectome: NDArray[np.float64],
    upper_triangle_only: bool = True,
) -> NDArray[np.float64]:
    """Flatten a 3D connectome array to 2D (subjects x features).

    Parameters
    ----------
    connectome : ndarray of shape (n_regions, n_regions, n_subjects)
        3D array of connectivity matrices.
    upper_triangle_only : bool, default=True
        If True, only use upper triangle (excluding diagonal) to avoid
        redundancy in symmetric matrices.

    Returns
    -------
    flattened : ndarray of shape (n_subjects, n_features)
        Flattened connectivity features.
    """
    n_regions = connectome.shape[0]
    n_subjects = connectome.shape[2]

    if upper_triangle_only:
        # Get indices for upper triangle
        triu_indices = np.triu_indices(n_regions, k=1)
        n_features = len(triu_indices[0])
        flattened = np.zeros((n_subjects, n_features))

        for i in range(n_subjects):
            flattened[i, :] = connectome[:, :, i][triu_indices]
    else:
        # Flatten entire matrix
        flattened = connectome.reshape(-1, n_subjects).T

    return flattened


def remove_zero_variance_features(
    X: NDArray[np.float64],
    threshold: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Remove features with zero or near-zero variance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    threshold : float, default=0.0
        Minimum variance threshold. Features with variance below this
        will be removed.

    Returns
    -------
    X_filtered : ndarray of shape (n_samples, n_filtered_features)
        Feature matrix with low-variance features removed.
    mask : ndarray of shape (n_features,)
        Boolean mask indicating which features were kept.
    """
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return X[:, mask], mask


def log_transform(
    X: NDArray[np.float64],
    offset: float = 1.0,
) -> NDArray[np.float64]:
    """Apply log transformation to handle skewed distributions.

    Uses log1p (log(x + offset)) to handle zero values.

    Parameters
    ----------
    X : ndarray
        Input array.
    offset : float, default=1.0
        Offset to add before log transform (for handling zeros).

    Returns
    -------
    X_transformed : ndarray
        Log-transformed array.
    """
    return np.log(X + offset)


def minmax_normalize(
    X: NDArray[np.float64],
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[NDArray[np.float64], float, float]:
    """Normalize features to a given range.

    Parameters
    ----------
    X : ndarray
        Input array.
    feature_range : tuple of (min, max), default=(0, 1)
        Desired range of transformed data.

    Returns
    -------
    X_normalized : ndarray
        Normalized array.
    data_min : float
        Minimum value of original data.
    data_max : float
        Maximum value of original data.
    """
    data_min = X.min()
    data_max = X.max()

    X_std = (X - data_min) / (data_max - data_min)
    X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]

    return X_scaled, data_min, data_max


def preprocess_connectome(
    connectome: NDArray[np.float64],
    log_transform_data: bool = True,
    normalize: bool = True,
    remove_zero_var: bool = True,
) -> tuple[NDArray[np.float64], dict]:
    """Full preprocessing pipeline for connectome data.

    Parameters
    ----------
    connectome : ndarray of shape (n_regions, n_regions, n_subjects)
        3D array of connectivity matrices.
    log_transform_data : bool, default=True
        Whether to apply log transformation.
    normalize : bool, default=True
        Whether to apply min-max normalization.
    remove_zero_var : bool, default=True
        Whether to remove zero-variance features.

    Returns
    -------
    X_processed : ndarray of shape (n_subjects, n_features)
        Processed feature matrix.
    metadata : dict
        Dictionary containing preprocessing parameters for inverse transform.
    """
    metadata = {}

    # Flatten to 2D
    X = flatten_connectome(connectome, upper_triangle_only=True)
    metadata["original_shape"] = connectome.shape

    # Remove zero-variance features
    if remove_zero_var:
        X, var_mask = remove_zero_variance_features(X)
        metadata["variance_mask"] = var_mask
        metadata["n_removed_features"] = (~var_mask).sum()

    # Log transform
    if log_transform_data:
        X = log_transform(X)
        metadata["log_transformed"] = True

    # Normalize
    if normalize:
        X, data_min, data_max = minmax_normalize(X)
        metadata["data_min"] = data_min
        metadata["data_max"] = data_max
        metadata["normalized"] = True

    return X, metadata
