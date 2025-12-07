"""Pytest fixtures for connectopy tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_connectome():
    """Generate a sample 3D connectome array for testing."""
    np.random.seed(42)
    n_regions = 68
    n_subjects = 50

    # Create symmetric positive matrices
    connectome = np.zeros((n_regions, n_regions, n_subjects))
    for i in range(n_subjects):
        mat = np.random.exponential(scale=100, size=(n_regions, n_regions))
        mat = (mat + mat.T) / 2  # Make symmetric
        np.fill_diagonal(mat, 0)  # No self-connections
        connectome[:, :, i] = mat

    return connectome


@pytest.fixture
def sample_subject_ids():
    """Generate sample subject IDs."""
    return np.arange(100001, 100051)


@pytest.fixture
def sample_traits_df():
    """Generate sample traits DataFrame."""
    np.random.seed(42)
    n_subjects = 50

    return pd.DataFrame(
        {
            "Subject": np.arange(100001, 100001 + n_subjects),
            "Gender": np.random.choice(["M", "F"], n_subjects),
            "Age": np.random.choice(["22-25", "26-30", "31-35", "36+"], n_subjects),
            "PMAT24_A_CR": np.random.randint(10, 30, n_subjects),
            "Strength_Unadj": np.random.normal(80, 20, n_subjects),
        }
    )


@pytest.fixture
def sample_pca_data():
    """Generate sample data with PC columns."""
    np.random.seed(42)
    n_subjects = 50
    n_components = 60

    data = {
        "Subject": np.arange(100001, 100001 + n_subjects),
        "Gender": np.random.choice(["M", "F"], n_subjects),
    }

    # Add PC columns with some gender effect on PC1
    for i in range(n_components):
        if i == 0:
            # Add a gender effect to PC1
            data[f"PC{i + 1}"] = np.where(
                np.array(data["Gender"]) == "M",
                np.random.normal(0.5, 1, n_subjects),
                np.random.normal(-0.5, 1, n_subjects),
            )
        else:
            data[f"PC{i + 1}"] = np.random.normal(0, 1, n_subjects)

    return pd.DataFrame(data)


@pytest.fixture
def sample_flattened_connectome():
    """Generate sample flattened connectome data."""
    np.random.seed(42)
    n_subjects = 50
    n_features = 68 * 67 // 2  # Upper triangle

    return np.random.exponential(scale=1, size=(n_subjects, n_features))
