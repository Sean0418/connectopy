"""Data loader for HCP connectome data.

This module provides functionality to load and merge structural connectome,
functional connectome, TNPCA coefficients, and trait data from the Human
Connectome Project.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.io import loadmat

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConnectomeDataLoader:
    """Load and manage HCP connectome data.

    This class handles loading of structural connectome (SC), functional
    connectome (FC), TNPCA coefficients, and trait data from .mat and .csv files.

    Parameters
    ----------
    data_dir : str or Path
        Path to the data directory containing raw/ and processed/ subdirectories.

    Attributes
    ----------
    data_dir : Path
        Base data directory path.
    raw_dir : Path
        Path to raw data directory.
    processed_dir : Path
        Path to processed data directory.

    Examples
    --------
    >>> loader = ConnectomeDataLoader("data/")
    >>> sc_data = loader.load_structural_connectome()
    >>> merged = loader.load_merged_dataset()
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Initialize the data loader.

        Parameters
        ----------
        data_dir : str or Path
            Path to the data directory.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    def load_structural_connectome(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Load raw structural connectome matrices.

        Returns
        -------
        connectome : ndarray of shape (n_regions, n_regions, n_subjects)
            3D array of structural connectivity matrices.
        subject_ids : ndarray of shape (n_subjects,)
            Array of subject IDs.

        Raises
        ------
        FileNotFoundError
            If the SC data file is not found.
        """
        sc_path = self.raw_dir / "SC" / "HCP_cortical_DesikanAtlas_SC.mat"
        if not sc_path.exists():
            raise FileNotFoundError(f"Structural connectome file not found: {sc_path}")

        data = loadmat(str(sc_path))
        connectome = data["hcp_sc_count"]
        subject_ids = np.squeeze(data["all_id"])

        return connectome, subject_ids

    def load_functional_connectome(self) -> tuple[list[NDArray[np.float64]], NDArray[np.int64]]:
        """Load raw functional connectome matrices.

        Returns
        -------
        connectome : list of ndarray
            List of functional connectivity matrices per subject.
        subject_ids : ndarray of shape (n_subjects,)
            Array of subject IDs.

        Raises
        ------
        FileNotFoundError
            If the FC data file is not found.
        """
        fc_path = self.raw_dir / "FC" / "HCP_cortical_DesikanAtlas_FC.mat"
        if not fc_path.exists():
            raise FileNotFoundError(f"Functional connectome file not found: {fc_path}")

        data = loadmat(str(fc_path))
        connectome = data["hcp_cortical_fc"]
        subject_ids = np.squeeze(data["subj_list"])

        return connectome, subject_ids

    def load_tnpca_structural(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Load TNPCA coefficients for structural connectome.

        Returns
        -------
        coefficients : ndarray of shape (n_subjects, n_components)
            PCA coefficient matrix (typically 60 components).
        subject_ids : ndarray of shape (n_subjects,)
            Array of subject IDs.

        Raises
        ------
        FileNotFoundError
            If the TNPCA file is not found.
        """
        tnpca_path = self.raw_dir / "TNPCA_Result" / "TNPCA_Coeff_HCP_Structural_Connectome.mat"
        if not tnpca_path.exists():
            raise FileNotFoundError(f"TNPCA structural file not found: {tnpca_path}")

        data = loadmat(str(tnpca_path))
        coefficients = data["PCA_Coeff"]
        subject_ids = np.squeeze(data["sub_id"])

        # Reshape to (n_subjects, n_components)
        n_subjects = len(subject_ids)
        coefficients = coefficients.reshape(n_subjects, -1)

        return coefficients, subject_ids

    def load_tnpca_functional(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Load TNPCA coefficients for functional connectome.

        Returns
        -------
        coefficients : ndarray of shape (n_subjects, n_components)
            PCA coefficient matrix (typically 60 components).
        subject_ids : ndarray of shape (n_subjects,)
            Array of subject IDs.

        Raises
        ------
        FileNotFoundError
            If the TNPCA file is not found.
        """
        tnpca_path = self.raw_dir / "TNPCA_Result" / "TNPCA_Coeff_HCP_Functional_Connectome.mat"
        if not tnpca_path.exists():
            raise FileNotFoundError(f"TNPCA functional file not found: {tnpca_path}")

        data = loadmat(str(tnpca_path))
        coefficients = data["PCA_Coeff"]
        subject_ids = np.squeeze(data["network_subject_ids"])

        # Reshape to (n_subjects, n_components)
        n_subjects = len(subject_ids)
        coefficients = coefficients.reshape(n_subjects, -1)

        return coefficients, subject_ids

    def load_traits(self) -> pd.DataFrame:
        """Load and merge trait data from CSV files.

        Returns
        -------
        traits : DataFrame
            Merged trait data with Subject as index.

        Raises
        ------
        FileNotFoundError
            If trait files are not found.
        """
        traits1_path = self.raw_dir / "traits" / "table1_hcp.csv"
        traits2_path = self.raw_dir / "traits" / "table2_hcp.csv"

        if not traits1_path.exists():
            raise FileNotFoundError(f"Traits file not found: {traits1_path}")
        if not traits2_path.exists():
            raise FileNotFoundError(f"Traits file not found: {traits2_path}")

        traits1 = pd.read_csv(traits1_path)
        traits2 = pd.read_csv(traits2_path)

        # Merge on Subject column
        traits = traits1.merge(traits2, on="Subject", how="inner")

        return traits

    def load_merged_dataset(
        self,
        include_raw_pca: bool = False,
        include_vae: bool = False,
    ) -> pd.DataFrame:
        """Load and merge all data sources into a single DataFrame.

        This combines TNPCA coefficients for both structural and functional
        connectomes with trait data.

        Parameters
        ----------
        include_raw_pca : bool, default=False
            Whether to include raw PCA scores (from pca-vae analysis).
        include_vae : bool, default=False
            Whether to include VAE latent dimensions (from pca-vae analysis).

        Returns
        -------
        merged : DataFrame
            Merged dataset with all features and traits.
        """
        # Load TNPCA coefficients
        struct_coeffs, struct_ids = self.load_tnpca_structural()
        func_coeffs, func_ids = self.load_tnpca_functional()

        # Create DataFrames
        struct_df = pd.DataFrame(
            struct_coeffs,
            columns=[f"Struct_PC{i + 1}" for i in range(struct_coeffs.shape[1])],
        )
        struct_df["Subject"] = struct_ids

        func_df = pd.DataFrame(
            func_coeffs,
            columns=[f"Func_PC{i + 1}" for i in range(func_coeffs.shape[1])],
        )
        func_df["Subject"] = func_ids

        # Load traits
        traits = self.load_traits()

        # Merge all data (inner join keeps only subjects present in all datasets)
        merged = traits.merge(struct_df, on="Subject", how="inner")
        merged = merged.merge(func_df, on="Subject", how="inner")

        # Optionally add processed data
        if include_raw_pca:
            raw_pca_path = self.processed_dir / "raw_pca_df.csv"
            if raw_pca_path.exists():
                raw_pca = pd.read_csv(raw_pca_path)
                merged = merged.merge(raw_pca, on="Subject", how="inner")

        if include_vae:
            vae_path = self.processed_dir / "vae_df.csv"
            if vae_path.exists():
                vae_df = pd.read_csv(vae_path)
                merged = merged.merge(vae_df, on="Subject", how="inner")

        return merged

    def save_merged_dataset(self, df: pd.DataFrame, filename: str = "full_data.csv") -> Path:
        """Save merged dataset to processed directory.

        Parameters
        ----------
        df : DataFrame
            Dataset to save.
        filename : str, default="full_data.csv"
            Output filename.

        Returns
        -------
        output_path : Path
            Path to saved file.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        return output_path
