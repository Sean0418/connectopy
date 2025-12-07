"""HCP data loader for brain connectome analysis.

This module provides functions to load and merge HCP data including:
- Structural connectivity (SC) matrices and PCA coefficients
- Functional connectivity (FC) matrices and PCA coefficients
- Cognitive trait measures from NIH Toolbox
- Alcohol and substance use measures from SSAGA
- Demographic information (age, sex)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional scipy for .mat file loading
try:
    from scipy import io as sio

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def load_mat_file(filepath: str | Path) -> dict[str, Any]:
    """Load a MATLAB .mat file.

    Args:
        filepath: Path to the .mat file.

    Returns
    -------
        Dictionary containing the variables from the .mat file.

    Raises
    ------
        ImportError: If scipy is not installed.
        FileNotFoundError: If the file doesn't exist.
    """
    if not SCIPY_AVAILABLE:
        msg = "scipy is required to load .mat files. Install with: pip install scipy"
        raise ImportError(msg)

    filepath = Path(filepath)
    if not filepath.exists():
        msg = f"File not found: {filepath}"
        raise FileNotFoundError(msg)

    result: dict[str, Any] = sio.loadmat(str(filepath), simplify_cells=True)
    return result


def load_structural_connectivity(
    data_dir: str | Path,
    load_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load structural connectivity data.

    Args:
        data_dir: Path to the data/raw directory.
        load_raw: If True, load raw 68x68 matrices. If False, load PCA coefficients.

    Returns
    -------
        Tuple of (subject_ids, connectivity_data).
        If load_raw=True: connectivity_data is (68, 68, n_subjects).
        If load_raw=False: connectivity_data is (n_subjects, n_components).
    """
    data_dir = Path(data_dir)

    if load_raw:
        mat_data = load_mat_file(data_dir / "SC" / "HCP_cortical_DesikanAtlas_SC.mat")
        subject_ids = mat_data.get("all_id", mat_data.get("subj_list", np.array([])))
        sc_data = mat_data.get("hcp_sc_count", mat_data.get("hcp_cortical_sc", np.array([])))
        return subject_ids, sc_data
    else:
        mat_data = load_mat_file(
            data_dir / "TNPCA_Result" / "TNPCA_Coeff_HCP_Structural_Connectome.mat"
        )
        subject_ids = mat_data.get("network_subject_ids", mat_data.get("subj_list", np.array([])))
        # PCA coefficients may be stored in different formats
        pca_coeff = mat_data.get("PCA_Coeff", mat_data.get("TNPCA_Coeff", np.array([])))

        # Handle different array shapes
        if pca_coeff.ndim == 3:
            # Shape (1, n_subjects, n_components) -> (n_subjects, n_components)
            pca_coeff = pca_coeff.squeeze()
        if pca_coeff.ndim == 1:
            pca_coeff = pca_coeff.reshape(-1, 1)

        return subject_ids, pca_coeff


def load_functional_connectivity(
    data_dir: str | Path,
    load_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load functional connectivity data.

    Args:
        data_dir: Path to the data/raw directory.
        load_raw: If True, load raw 68x68 matrices. If False, load PCA coefficients.

    Returns
    -------
        Tuple of (subject_ids, connectivity_data).
        If load_raw=True: connectivity_data is (68, 68, n_subjects).
        If load_raw=False: connectivity_data is (n_subjects, n_components).
    """
    data_dir = Path(data_dir)

    if load_raw:
        mat_data = load_mat_file(data_dir / "FC" / "HCP_cortical_DesikanAtlas_FC.mat")
        subject_ids = mat_data.get("subj_list", mat_data.get("all_id", np.array([])))
        fc_data = mat_data.get("hcp_cortical_fc", mat_data.get("hcp_fc", np.array([])))
        return subject_ids, fc_data
    else:
        mat_data = load_mat_file(
            data_dir / "TNPCA_Result" / "TNPCA_Coeff_HCP_Functional_Connectome.mat"
        )
        subject_ids = mat_data.get("network_subject_ids", mat_data.get("subj_list", np.array([])))
        pca_coeff = mat_data.get("PCA_Coeff", mat_data.get("TNPCA_Coeff", np.array([])))

        # Handle different array shapes
        if pca_coeff.ndim == 3:
            pca_coeff = pca_coeff.squeeze()
        if pca_coeff.ndim == 1:
            pca_coeff = pca_coeff.reshape(-1, 1)

        return subject_ids, pca_coeff


def load_cognitive_measures(data_dir: str | Path) -> pd.DataFrame:
    """Load cognitive measures from HCP data.

    This loads NIH Toolbox cognitive measures including:
    - Picture Sequence Memory (episodic memory)
    - Card Sort (executive function)
    - Flanker (attention/inhibition)
    - PMAT (fluid reasoning)
    - Reading (crystallized intelligence)
    - Picture Vocabulary (crystallized intelligence)
    - Processing Speed
    - List Sorting (working memory)

    Args:
        data_dir: Path to the data/raw directory.

    Returns
    -------
        DataFrame with subject IDs and cognitive measures.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "traits" / "table1_hcp.csv"

    if not csv_path.exists():
        msg = f"Cognitive data file not found: {csv_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)

    # Select cognitive columns
    cognitive_cols = [
        "Subject",
        "Gender",
        "Age",
        # NIH Toolbox measures (unadjusted scores)
        "PicSeq_Unadj",  # Picture Sequence Memory
        "CardSort_Unadj",  # Card Sort (Executive)
        "Flanker_Unadj",  # Flanker (Attention)
        "PMAT24_A_CR",  # Penn Matrix (Fluid Intelligence)
        "ReadEng_Unadj",  # Reading (Crystallized)
        "PicVocab_Unadj",  # Picture Vocabulary (Crystallized)
        "ProcSpeed_Unadj",  # Processing Speed
        "ListSort_Unadj",  # List Sorting (Working Memory)
        # Age-adjusted versions
        "PicSeq_AgeAdj",
        "CardSort_AgeAdj",
        "Flanker_AgeAdj",
        "ReadEng_AgeAdj",
        "PicVocab_AgeAdj",
        "ProcSpeed_AgeAdj",
        "ListSort_AgeAdj",
    ]

    # Filter to available columns
    available_cols = [c for c in cognitive_cols if c in df.columns]

    return pd.DataFrame(df[available_cols])


def load_alcohol_measures(data_dir: str | Path) -> pd.DataFrame:
    """Load alcohol and substance use measures from HCP data.

    This loads SSAGA (Semi-Structured Assessment for the Genetics of Alcoholism)
    measures including:
    - DSM-IV Alcohol Dependence diagnosis and symptoms
    - DSM-IV Alcohol Abuse diagnosis and symptoms
    - Drinking frequency and quantity measures
    - Age of first use

    Args:
        data_dir: Path to the data/raw directory.

    Returns
    -------
        DataFrame with subject IDs and alcohol measures.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "traits" / "table2_hcp.csv"

    if not csv_path.exists():
        msg = f"Alcohol data file not found: {csv_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)

    # Select alcohol-related columns
    alcohol_cols = [
        "Subject",
        # DSM-IV diagnoses
        "SSAGA_Alc_D4_Dp_Dx",  # Alcohol Dependence Diagnosis
        "SSAGA_Alc_D4_Dp_Sx",  # Alcohol Dependence Symptoms
        "SSAGA_Alc_D4_Ab_Dx",  # Alcohol Abuse Diagnosis
        "SSAGA_Alc_D4_Ab_Sx",  # Alcohol Abuse Symptoms
        # Drinking behavior (past 12 months)
        "SSAGA_Alc_12_Drinks_Per_Day",
        "SSAGA_Alc_12_Frq",  # Frequency
        "SSAGA_Alc_12_Frq_5plus",  # Binge frequency
        "SSAGA_Alc_12_Max_Drinks",
        # Heavy drinking period
        "SSAGA_Alc_Hvy_Drinks_Per_Day",
        "SSAGA_Alc_Hvy_Frq",
        "SSAGA_Alc_Hvy_Max_Drinks",
        # Age of first use
        "SSAGA_Alc_Age_1st_Use",
        # Recent drinking (past 7 days)
        "Total_Drinks_7days",
        "Num_Days_Drank_7days",
    ]

    # Filter to available columns
    available_cols = [c for c in alcohol_cols if c in df.columns]

    return pd.DataFrame(df[available_cols])


def load_merged_hcp_data(
    data_dir: str | Path,
    n_sc_components: int = 10,
    n_fc_components: int = 10,
) -> pd.DataFrame:
    """Load and merge all HCP data for mediation analysis.

    This function loads and merges:
    - Structural connectivity PCA coefficients
    - Functional connectivity PCA coefficients
    - Cognitive measures
    - Alcohol measures
    - Demographics (sex, age)

    Args:
        data_dir: Path to the data/raw directory.
        n_sc_components: Number of SC PCA components to include.
        n_fc_components: Number of FC PCA components to include.

    Returns
    -------
        Merged DataFrame with all variables aligned by subject ID.
    """
    data_dir = Path(data_dir)

    # Load PCA coefficients
    sc_ids, sc_pca = load_structural_connectivity(data_dir, load_raw=False)
    fc_ids, fc_pca = load_functional_connectivity(data_dir, load_raw=False)

    # Create DataFrames for network data
    sc_cols = [f"SC_PC{i + 1}" for i in range(min(n_sc_components, sc_pca.shape[1]))]
    sc_df = pd.DataFrame(sc_pca[:, : len(sc_cols)], columns=sc_cols)
    sc_df["Subject"] = sc_ids

    fc_cols = [f"FC_PC{i + 1}" for i in range(min(n_fc_components, fc_pca.shape[1]))]
    fc_df = pd.DataFrame(fc_pca[:, : len(fc_cols)], columns=fc_cols)
    fc_df["Subject"] = fc_ids

    # Load traits
    cognitive_df = load_cognitive_measures(data_dir)
    alcohol_df = load_alcohol_measures(data_dir)

    # Merge all data
    merged = cognitive_df.merge(alcohol_df, on="Subject", how="inner")
    merged = merged.merge(sc_df, on="Subject", how="inner")
    merged = merged.merge(fc_df, on="Subject", how="inner")

    # Clean up
    merged = merged.dropna(subset=["Gender"])  # Must have sex info

    print(f"Merged data: {len(merged)} subjects")
    print(f"SC components: {len(sc_cols)}, FC components: {len(fc_cols)}")
    print(f"Gender distribution: {merged['Gender'].value_counts().to_dict()}")

    return merged


def create_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create composite cognitive scores from individual measures.

    Creates:
    - FluidComposite: Average of PMAT, CardSort, Flanker, ListSort
    - CrystalComposite: Average of Reading, PicVocab
    - OverallCognitive: Average of all cognitive measures

    Args:
        df: DataFrame with individual cognitive measures.

    Returns
    -------
        DataFrame with added composite score columns.
    """
    df = df.copy()

    # Fluid intelligence composite
    fluid_cols = ["PMAT24_A_CR", "CardSort_Unadj", "Flanker_Unadj", "ListSort_Unadj"]
    available_fluid = [c for c in fluid_cols if c in df.columns]
    if available_fluid:
        # Z-score each measure before averaging
        fluid_z = df[available_fluid].apply(lambda x: (x - x.mean()) / x.std())
        df["FluidComposite"] = fluid_z.mean(axis=1)

    # Crystallized intelligence composite
    crystal_cols = ["ReadEng_Unadj", "PicVocab_Unadj"]
    available_crystal = [c for c in crystal_cols if c in df.columns]
    if available_crystal:
        crystal_z = df[available_crystal].apply(lambda x: (x - x.mean()) / x.std())
        df["CrystalComposite"] = crystal_z.mean(axis=1)

    # Overall cognitive composite
    all_cog_cols = available_fluid + available_crystal + ["PicSeq_Unadj", "ProcSpeed_Unadj"]
    available_all = [c for c in all_cog_cols if c in df.columns]
    if available_all:
        all_z = df[available_all].apply(lambda x: (x - x.mean()) / x.std())
        df["OverallCognitive"] = all_z.mean(axis=1)

    return df


def create_alcohol_severity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create a continuous alcohol severity score.

    Combines multiple alcohol measures into a single severity score:
    - Dependence diagnosis (weighted heavily)
    - Symptom counts
    - Drinking frequency/quantity

    Args:
        df: DataFrame with alcohol measures.

    Returns
    -------
        DataFrame with added AlcoholSeverity column.
    """
    df = df.copy()

    severity_components = []

    # Dependence diagnosis (binary, weighted x2)
    if "SSAGA_Alc_D4_Dp_Dx" in df.columns:
        dep_dx = df["SSAGA_Alc_D4_Dp_Dx"].fillna(0).clip(0, 1) * 2
        severity_components.append(dep_dx)

    # Dependence symptoms (0-7)
    if "SSAGA_Alc_D4_Dp_Sx" in df.columns:
        dep_sx: pd.Series[float] = df["SSAGA_Alc_D4_Dp_Sx"].fillna(0) / 7  # Normalize to 0-1
        severity_components.append(dep_sx)

    # Abuse diagnosis
    if "SSAGA_Alc_D4_Ab_Dx" in df.columns:
        ab_dx = df["SSAGA_Alc_D4_Ab_Dx"].fillna(0).clip(0, 1)
        severity_components.append(ab_dx)

    # Drinking frequency (normalized)
    if "SSAGA_Alc_12_Frq" in df.columns:
        frq = df["SSAGA_Alc_12_Frq"].fillna(0)
        frq_norm = (frq - frq.min()) / (frq.max() - frq.min() + 1e-10)
        severity_components.append(frq_norm)

    if severity_components:
        df["AlcoholSeverity"] = sum(severity_components) / len(severity_components)
    else:
        df["AlcoholSeverity"] = 0.0

    return df
