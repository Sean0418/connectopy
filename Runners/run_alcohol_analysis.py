#!/usr/bin/env python3
"""Alcohol Use Disorder Classification Analysis.

This script runs Random Forest and Explainable Boosting Machine (EBM) classifiers
to predict alcohol use disorder from brain connectome and cognitive features.

The analysis is stratified by sex and runs multiple feature set variants:
- Cognitive features only
- Connectome features only (TNPCA, VAE, or raw PCA)
- Combined cognitive + connectome features

Usage:
    python Runners/run_alcohol_analysis.py [OPTIONS]

Options:
    --data-path PATH       Path to full_data.csv (default: data/processed/full_data.csv)
    --output-dir PATH      Output directory (default: output/alcohol_analysis)
    --variants VARIANTS    Comma-separated list of variants (default: tnpca,vae,pca)
    --model-types TYPES    Comma-separated model types (default: rf,ebm)
    --skip-plots           Skip generating plots
    --verbose              Print detailed progress

Based on analysis from the yinyu branch.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# REPRODUCIBILITY: Global random seed
# ============================================================================
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def prepare_alcohol_target(df):
    """Create binary alcohol use disorder target from HCP data.

    HCP coding for SSAGA_Alc_D4_Ab_Dx:
    - 1 = No diagnosis
    - 5 = Yes diagnosis (alcohol abuse/dependence)

    Parameters
    ----------
    df : DataFrame
        Data containing SSAGA_Alc_D4_Ab_Dx column.

    Returns
    -------
    df : DataFrame
        Data with added 'alc_y' column (1=positive, 0=negative/NA).
    """
    import numpy as np

    if "SSAGA_Alc_D4_Ab_Dx" not in df.columns:
        raise ValueError("Column 'SSAGA_Alc_D4_Ab_Dx' not found in data")

    label_raw = df["SSAGA_Alc_D4_Ab_Dx"]
    df = df.copy()
    df["alc_y"] = np.where(label_raw == 5, 1, 0).astype(int)

    return df


def prepare_ebm_input_datasets(df, output_dir: Path):
    """Create variant-specific datasets for EBM/RF analysis.

    Creates three datasets combining cognitive features with different
    connectome representations:
    - tnpca: TN-PCA structural and functional connectome scores
    - vae: VAE latent dimensions
    - pca: Raw PCA scores

    Parameters
    ----------
    df : DataFrame
        Full dataset with all features.
    output_dir : Path
        Directory to save the variant datasets.

    Returns
    -------
    datasets : dict
        Dictionary mapping variant name to DataFrame.
    """
    from connectopy.models import get_cognitive_features

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get cognitive features
    cog_features = get_cognitive_features(df, include_age=True)
    print(f"  Cognitive features found: {len(cog_features)}")

    # Common columns
    common_cols = ["Subject", "Gender", "alc_y"]
    common_cols = [c for c in common_cols if c in df.columns]

    # Define connectome feature sets
    struct_tnpca = [f"Struct_PC{i}" for i in range(1, 61) if f"Struct_PC{i}" in df.columns]
    func_tnpca = [f"Func_PC{i}" for i in range(1, 61) if f"Func_PC{i}" in df.columns]

    struct_vae = [f"VAE_Struct_LD{i}" for i in range(1, 61) if f"VAE_Struct_LD{i}" in df.columns]
    func_vae = [f"VAE_Func_LD{i}" for i in range(1, 61) if f"VAE_Func_LD{i}" in df.columns]

    struct_pca = [f"Raw_Struct_PC{i}" for i in range(1, 61) if f"Raw_Struct_PC{i}" in df.columns]
    func_pca = [f"Raw_Func_PC{i}" for i in range(1, 61) if f"Raw_Func_PC{i}" in df.columns]

    print(f"  TN-PCA features: {len(struct_tnpca) + len(func_tnpca)}")
    print(f"  VAE features: {len(struct_vae) + len(func_vae)}")
    print(f"  Raw PCA features: {len(struct_pca) + len(func_pca)}")

    datasets = {}

    # Build variant datasets
    def build_dataset(feats_list):
        feats = common_cols + cog_features + feats_list
        feats = list(dict.fromkeys(feats))  # Remove duplicates, preserve order
        return df[feats].copy()

    datasets["tnpca"] = build_dataset(struct_tnpca + func_tnpca)
    datasets["vae"] = build_dataset(struct_vae + func_vae)
    datasets["pca"] = build_dataset(struct_pca + func_pca)

    # Save datasets
    for name, ds in datasets.items():
        out_path = output_dir / f"ebm_input_{name}.csv"
        ds.to_csv(out_path, index=False)
        print(f"  Saved {name} dataset: {ds.shape} to {out_path}")

    return datasets


def train_rf_variant(
    df_sex,
    feature_names: list[str],
    sex_label: str,
    variant_label: str,
    output_dir: Path,
    verbose: bool = True,
):
    """Train Random Forest for one sex/variant combination.

    Parameters
    ----------
    df_sex : DataFrame
        Data for one sex.
    feature_names : list of str
        Feature columns to use.
    sex_label : str
        'M' or 'F'.
    variant_label : str
        E.g., 'tnpca', 'cog_only', etc.
    output_dir : Path
        Directory for outputs.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    metrics : dict or None
        Training metrics if successful.
    """
    import numpy as np

    from connectopy.models import ConnectomeRandomForest

    cols = feature_names + ["alc_y"]
    sub = df_sex[cols].dropna().copy()

    if sub.empty or len(sub) < 50:
        if verbose:
            print(f"    [WARN] Insufficient data for {sex_label}/{variant_label}")
        return None

    X = sub[feature_names].values
    y = sub["alc_y"].astype(int).values

    if len(np.unique(y)) < 2:
        if verbose:
            print(f"    [WARN] Only one class present for {sex_label}/{variant_label}")
        return None

    if verbose:
        print(f"\n  Training RF: sex={sex_label}, variant={variant_label}")
        print(f"    Samples: {len(y)}, Features: {len(feature_names)}")
        print(f"    Positive rate: {y.mean():.3f}")

    # Train with CV
    clf = ConnectomeRandomForest(
        n_estimators=500, class_weight="balanced", random_state=RANDOM_SEED
    )
    metrics = clf.fit_with_cv(
        X,
        y,
        feature_names=feature_names,
        handle_imbalance=True,
    )

    metrics["sex"] = sex_label
    metrics["variant"] = variant_label

    if verbose:
        print(f"    CV AUC: {metrics['cv_best_auc']:.3f}")
        print(f"    Test AUC: {metrics['test_auc']:.3f}")

    # Save model
    import joblib

    model_path = output_dir / "models" / f"rf_{sex_label}_{variant_label}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": feature_names}, model_path)

    # Save ROC curve plot
    _save_roc_plot(clf, sex_label, variant_label, "RF", output_dir)

    # Save feature importance plot
    _save_importance_plot(clf, sex_label, variant_label, "RF", output_dir)

    return metrics


def train_ebm_variant(
    df_sex,
    feature_names: list[str],
    sex_label: str,
    variant_label: str,
    output_dir: Path,
    verbose: bool = True,
):
    """Train EBM for one sex/variant combination.

    Parameters
    ----------
    df_sex : DataFrame
        Data for one sex.
    feature_names : list of str
        Feature columns to use.
    sex_label : str
        'M' or 'F'.
    variant_label : str
        E.g., 'tnpca', 'cog_only', etc.
    output_dir : Path
        Directory for outputs.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    metrics : dict or None
        Training metrics if successful.
    """
    import numpy as np

    from connectopy.models import ConnectomeEBM

    cols = feature_names + ["alc_y"]
    sub = df_sex[cols].dropna().copy()

    if sub.empty or len(sub) < 50:
        if verbose:
            print(f"    [WARN] Insufficient data for {sex_label}/{variant_label}")
        return None

    X = sub[feature_names].values
    y = sub["alc_y"].astype(int).values

    if len(np.unique(y)) < 2:
        if verbose:
            print(f"    [WARN] Only one class present for {sex_label}/{variant_label}")
        return None

    if verbose:
        print(f"\n  Training EBM: sex={sex_label}, variant={variant_label}")
        print(f"    Samples: {len(y)}, Features: {len(feature_names)}")
        print(f"    Positive rate: {y.mean():.3f}")

    # Train with CV
    try:
        clf = ConnectomeEBM(
            max_bins=32,
            learning_rate=0.01,
            max_leaves=3,
            interactions=0,
            random_state=RANDOM_SEED,
        )
        metrics = clf.fit_with_cv(
            X,
            y,
            feature_names=feature_names,
            handle_imbalance=True,
        )
    except ImportError:
        if verbose:
            print("    [ERROR] interpret package not installed, skipping EBM")
        return None

    metrics["sex"] = sex_label
    metrics["variant"] = variant_label

    if verbose:
        print(f"    CV AUC: {metrics['cv_best_auc']:.3f}")
        print(f"    Test AUC: {metrics['test_auc']:.3f}")

    # Save model
    import joblib

    model_path = output_dir / "models" / f"ebm_{sex_label}_{variant_label}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": feature_names}, model_path)

    # Save ROC curve plot
    _save_roc_plot(clf, sex_label, variant_label, "EBM", output_dir)

    # Save feature importance plot
    _save_importance_plot(clf, sex_label, variant_label, "EBM", output_dir)

    return metrics


def _save_roc_plot(clf, sex_label: str, variant_label: str, model_type: str, output_dir: Path):
    """Save ROC curve plot for a trained model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        roc_data = clf.get_roc_data()
    except (ValueError, AttributeError):
        return

    plt.figure(figsize=(6, 5))
    plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"AUC={roc_data['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_type} ROC\nsex={sex_label}, variant={variant_label}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plots_dir = output_dir / "plots" / "roc"
    plots_dir.mkdir(parents=True, exist_ok=True)
    roc_path = plots_dir / f"roc_{model_type}_{sex_label}_{variant_label}.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()


def _save_importance_plot(
    clf, sex_label: str, variant_label: str, model_type: str, output_dir: Path, top_k: int = 20
):
    """Save feature importance plot for a trained model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if clf.feature_importances_ is None:
        return

    topk = clf.feature_importances_.head(top_k)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=topk["Importance"].values, y=topk["Feature"].values, orient="h")
    plt.xlabel("Feature Importance")
    plt.ylabel("")
    plt.title(f"{model_type} Feature Importance\nsex={sex_label}, variant={variant_label}")
    plt.tight_layout()

    plots_dir = output_dir / "plots" / "importance"
    plots_dir.mkdir(parents=True, exist_ok=True)
    imp_path = plots_dir / f"importance_{model_type}_{sex_label}_{variant_label}.png"
    plt.savefig(imp_path, dpi=150)
    plt.close()


def run_alcohol_analysis(
    data_path: Path,
    output_dir: Path,
    variants: list[str] | None = None,
    model_types: list[str] | None = None,
    verbose: bool = True,
):
    """Run the complete alcohol classification analysis.

    Parameters
    ----------
    data_path : Path
        Path to full_data.csv.
    output_dir : Path
        Output directory.
    variants : list of str, optional
        Connectome variants to use. Defaults to ['tnpca', 'vae', 'pca'].
    model_types : list of str, optional
        Model types to train ('rf', 'ebm'). Defaults to ['rf', 'ebm'].
    verbose : bool
        Whether to print progress.
    """
    import pandas as pd

    from connectopy.models import get_cognitive_features, get_connectome_features

    # Set defaults for mutable arguments
    if variants is None:
        variants = ["tnpca", "vae", "pca"]
    if model_types is None:
        model_types = ["rf", "ebm"]

    print("=" * 60)
    print("Alcohol Use Disorder Classification Analysis")
    print("=" * 60)

    # Load data
    print("\nStep 1: Loading Data")
    print("-" * 40)
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure full_data.csv exists in data/processed/")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} subjects, {len(df.columns)} columns")

    # Prepare target
    print("\nStep 2: Preparing Target Variable")
    print("-" * 40)
    df = prepare_alcohol_target(df)
    print("  alc_y distribution:")
    print(f"    0 (no diagnosis): {(df['alc_y'] == 0).sum()}")
    print(f"    1 (alcohol abuse/dependence): {(df['alc_y'] == 1).sum()}")

    # Check for required columns
    if "Gender" not in df.columns:
        print("ERROR: 'Gender' column not found")
        sys.exit(1)

    # Prepare variant datasets
    print("\nStep 3: Preparing Feature Datasets")
    print("-" * 40)
    ebm_input_dir = output_dir / "ebm_inputs"
    datasets = prepare_ebm_input_datasets(df, ebm_input_dir)

    # Run analysis
    print("\nStep 4: Training Models")
    print("-" * 40)

    all_results = []

    for variant in variants:
        if variant not in datasets:
            print(f"  [WARN] Variant {variant} not available, skipping")
            continue

        print(f"\n{'=' * 50}")
        print(f"Variant: {variant}")
        print("=" * 50)

        df_var = datasets[variant]

        # Get feature sets
        cog_features = get_cognitive_features(df_var)
        conn_features = get_connectome_features(df_var, variant)

        # Define feature set variants
        feature_sets = {}

        # Full: cognitive + connectome
        all_feats = [c for c in df_var.columns if c not in {"Subject", "Gender", "alc_y"}]
        feature_sets[variant] = all_feats

        # Cognitive only (only from tnpca to avoid duplication)
        if variant == "tnpca":
            feature_sets["cog_only"] = cog_features

        # Connectome only
        feature_sets[f"{variant}_only"] = conn_features

        print(f"  Cognitive features: {len(cog_features)}")
        print(f"  Connectome features ({variant}): {len(conn_features)}")

        for sex in ["M", "F"]:
            df_sex = df_var[df_var["Gender"] == sex].copy()
            if df_sex.empty:
                continue

            pos_rate = df_sex["alc_y"].mean()
            print(f"\n  Sex={sex}, positive rate: {pos_rate:.3f}")

            for variant_label, feats in feature_sets.items():
                if len(feats) == 0:
                    print(f"    [WARN] No features for {variant_label}, skipping")
                    continue

                # Train RF
                if "rf" in model_types:
                    try:
                        metrics = train_rf_variant(
                            df_sex, feats, sex, variant_label, output_dir, verbose
                        )
                        if metrics:
                            metrics["model"] = "RF"
                            all_results.append(metrics)
                    except Exception as e:
                        print(f"    [ERROR] RF {sex}/{variant_label}: {e}")

                # Train EBM
                if "ebm" in model_types:
                    try:
                        metrics = train_ebm_variant(
                            df_sex, feats, sex, variant_label, output_dir, verbose
                        )
                        if metrics:
                            metrics["model"] = "EBM"
                            all_results.append(metrics)
                    except Exception as e:
                        print(f"    [ERROR] EBM {sex}/{variant_label}: {e}")

    # Save summary metrics
    print("\nStep 5: Saving Results")
    print("-" * 40)

    if all_results:
        results_df = pd.DataFrame(all_results)

        # Reorder columns
        first_cols = ["model", "sex", "variant", "n_train", "n_test", "cv_best_auc", "test_auc"]
        other_cols = [c for c in results_df.columns if c not in first_cols]
        col_order = [c for c in first_cols if c in results_df.columns] + other_cols
        results_df = results_df[col_order]  # type: ignore[assignment]

        summary_path = output_dir / "alcohol_classification_summary.csv"
        results_df.to_csv(summary_path, index=False)
        print(f"  Saved summary metrics to {summary_path}")

        # Print summary table
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        summary = results_df[
            ["model", "sex", "variant", "cv_best_auc", "test_auc", "test_accuracy"]
        ]
        print(summary.to_string(index=False))

    else:
        print("  No models were successfully trained.")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - alcohol_classification_summary.csv")
    print("  - ebm_inputs/ (variant datasets)")
    print("  - models/ (trained model files)")
    print("  - plots/roc/ (ROC curves)")
    print("  - plots/importance/ (feature importance)")


def main():
    """Run the alcohol analysis from command line."""
    parser = argparse.ArgumentParser(
        description="Run alcohol use disorder classification analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Runners/run_alcohol_analysis.py                          # Run full analysis
  python Runners/run_alcohol_analysis.py --model-types rf         # RF only
  python Runners/run_alcohol_analysis.py --variants tnpca         # TNPCA variant only
  python Runners/run_alcohol_analysis.py --skip-plots             # Skip plot generation
        """,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to full_data.csv (default: data/processed/full_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/alcohol_analysis)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="tnpca,vae,pca",
        help="Comma-separated list of variants (default: tnpca,vae,pca)",
    )
    parser.add_argument(
        "--model-types",
        type=str,
        default="rf,ebm",
        help="Comma-separated model types (default: rf,ebm)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )
    args = parser.parse_args()

    # Set defaults
    project_root = get_project_root()

    if args.data_path is None:
        data_path = project_root / "data" / "processed" / "full_data.csv"
    else:
        data_path = Path(args.data_path)

    if args.output_dir is None:
        output_dir = project_root / "output" / "alcohol_analysis"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",")]
    model_types = [m.strip() for m in args.model_types.split(",")]

    run_alcohol_analysis(
        data_path=data_path,
        output_dir=output_dir,
        variants=variants,
        model_types=model_types,
        skip_plots=args.skip_plots,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
