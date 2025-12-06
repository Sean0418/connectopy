#!/usr/bin/env python3
"""Brain Connectome Analysis Pipeline Runner.

Platform-independent script to run the complete analysis pipeline.
Automatically sets up virtual environment and installs dependencies.

Usage:
    python Runners/run_pipeline.py [OPTIONS]

Options:
    --skip-pca           Skip PCA analysis
    --skip-vae           Skip VAE analysis
    --skip-dimorphism    Skip dimorphism analysis
    --skip-ml            Skip ML classification
    --skip-plots         Skip visualization generation
    --quick              Run quick mode (skip PCA, VAE, and plots)

The pipeline includes:
    1. Data Loading - Load and merge HCP data
    2. PCA Analysis - Dimensionality reduction on raw connectomes
    3. VAE Analysis - Variational autoencoder for nonlinear embeddings
    4. Dimorphism Analysis - Sexual dimorphism statistical tests
    5. ML Classification - Train gender classifier
    6. Visualization - Generate analysis plots
"""

import argparse
import sys
from pathlib import Path

# Handle imports for both direct execution and module import
try:
    from Runners.setup_environment import relaunch_in_venv, setup_environment
except ImportError:
    from setup_environment import relaunch_in_venv, setup_environment  # type: ignore[no-redef]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def check_data_exists(data_dir: Path) -> bool:
    """Check if required data files exist."""
    required_files = [
        data_dir / "raw" / "TNPCA_Result" / "TNPCA_Coeff_HCP_Structural_Connectome.mat",
        data_dir / "raw" / "TNPCA_Result" / "TNPCA_Coeff_HCP_Functional_Connectome.mat",
        data_dir / "raw" / "traits" / "table1_hcp.csv",
        data_dir / "raw" / "traits" / "table2_hcp.csv",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("Missing required data files:")
        for f in missing:
            print(f"  - {f}")
        return False
    return True


def check_raw_connectomes_exist(data_dir: Path) -> bool:
    """Check if raw connectome data exists for PCA/VAE analysis."""
    sc_path = data_dir / "raw" / "SC"
    fc_path = data_dir / "raw" / "FC"
    return sc_path.exists() or fc_path.exists()


def run_pca_analysis(data_dir: Path, output_dir: Path) -> None:
    """Run PCA on raw connectome data."""
    import numpy as np
    from scipy.io import loadmat

    from brain_connectome.analysis import ConnectomePCA

    print("Loading raw structural connectome data...")
    sc_path = data_dir / "raw" / "SC" / "HCP_cortical_DesikanAtlas_SC.mat"

    if not sc_path.exists():
        print(f"  Raw SC data not found at {sc_path}")
        print("  Using pre-computed TNPCA coefficients instead.")
        return

    mat = loadmat(str(sc_path))

    # Look specifically for the connectome data key
    sc_data = None

    # Known key names in HCP data
    connectome_keys = ["hcp_sc_count", "sc_data", "SC", "connectome"]

    for key in connectome_keys:
        if key in mat:
            sc_data = mat[key]
            break

    if sc_data is None:
        # Fallback: find 3D array
        for key, arr in mat.items():
            if not key.startswith("_") and hasattr(arr, "shape") and len(arr.shape) == 3:
                sc_data = arr
                break

    if sc_data is None:
        print("  Could not find connectome data in .mat file")
        return

    # Flatten 3D connectome data (regions x regions x subjects)
    if len(sc_data.shape) == 3:
        n_regions = sc_data.shape[0]
        n_subjects = sc_data.shape[2]
        # Flatten upper triangle of each subject's connectome
        triu_idx = np.triu_indices(n_regions, k=1)
        X = np.zeros((n_subjects, len(triu_idx[0])))
        for i in range(n_subjects):
            X[i, :] = sc_data[:, :, i][triu_idx]
        n_features = len(triu_idx[0])
        print(f"  Loaded {n_subjects} subjects, {n_features} features")
        print(f"  (upper triangle of {n_regions}x{n_regions} connectome)")
    else:
        X = sc_data
        print(f"  Loaded data shape: {X.shape}")

    # Run PCA
    print("Running PCA with 60 components...")
    pca = ConnectomePCA(n_components=60, scale=True)
    pca.fit(X)

    print(f"  Total variance explained: {pca.total_variance_explained_:.2%}")

    # Save variance report
    variance_report = pca.get_variance_report()
    variance_report.to_csv(output_dir / "pca_variance.csv", index=False)
    print(f"  Saved variance report to {output_dir / 'pca_variance.csv'}")

    # Save PC scores
    pc_scores = pca.to_dataframe(X, prefix="SC_PC")
    pc_scores.to_csv(output_dir / "pca_scores.csv", index=False)
    print(f"  Saved PC scores to {output_dir / 'pca_scores.csv'}")


def run_vae_analysis(data_dir: Path, output_dir: Path, epochs: int = 200) -> None:
    """Run VAE on raw connectome data."""
    import numpy as np
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split

    from brain_connectome.analysis import ConnectomeVAE

    print("Loading raw structural connectome data...")
    sc_path = data_dir / "raw" / "SC" / "HCP_cortical_DesikanAtlas_SC.mat"

    if not sc_path.exists():
        print(f"  Raw SC data not found at {sc_path}")
        print("  Skipping VAE analysis (requires raw connectome data).")
        return

    mat = loadmat(str(sc_path))

    # Look specifically for the connectome data key
    sc_data = None
    connectome_keys = ["hcp_sc_count", "sc_data", "SC", "connectome"]

    for key in connectome_keys:
        if key in mat:
            sc_data = mat[key]
            break

    if sc_data is None:
        # Fallback: find 3D array
        for key, arr in mat.items():
            if not key.startswith("_") and hasattr(arr, "shape") and len(arr.shape) == 3:
                sc_data = arr
                break

    if sc_data is None:
        print("  Could not find connectome data in .mat file")
        return

    # Flatten 3D connectome data (regions x regions x subjects)
    if len(sc_data.shape) == 3:
        n_regions = sc_data.shape[0]
        n_subjects = sc_data.shape[2]
        triu_idx = np.triu_indices(n_regions, k=1)
        X = np.zeros((n_subjects, len(triu_idx[0])))
        for i in range(n_subjects):
            X[i, :] = sc_data[:, :, i][triu_idx]
        print(f"  Loaded {n_subjects} subjects, {len(triu_idx[0])} features")
    else:
        X = sc_data
        print(f"  Loaded data shape: {X.shape}")

    # Normalize
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/val split
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

    print(f"Training VAE for {epochs} epochs...")
    print(f"  Training: {len(X_train)} subjects, Validation: {len(X_val)} subjects")
    vae = ConnectomeVAE(latent_dim=60, hidden_dim=256, dropout=0.2)
    vae.fit(X_train, X_val, epochs=epochs, verbose=True)

    # Save latent representations
    latent = vae.to_dataframe(X_scaled, prefix="VAE_LD")
    latent.to_csv(output_dir / "vae_latent.csv", index=False)
    print(f"  Saved VAE latent representations to {output_dir / 'vae_latent.csv'}")

    # Save training history
    import pandas as pd

    history = pd.DataFrame(
        {
            "Epoch": range(1, len(vae.train_losses) + 1),
            "Train_Loss": vae.train_losses,
            "Val_Loss": vae.val_losses
            if vae.val_losses
            else [float("nan")] * len(vae.train_losses),
        }
    )
    history.to_csv(output_dir / "vae_training_history.csv", index=False)
    print(f"  Saved training history to {output_dir / 'vae_training_history.csv'}")


def run_dimorphism_analysis(data, output_dir: Path):
    """Run sexual dimorphism analysis."""
    from brain_connectome import DimorphismAnalysis

    dimorphism_output = output_dir / "dimorphism_results.csv"

    print("Running dimorphism analysis on structural PCs...")
    analysis = DimorphismAnalysis(data)
    struct_pcs = [f"Struct_PC{i}" for i in range(1, 61)]

    # Filter to existing columns
    existing_pcs = [c for c in struct_pcs if c in data.columns]
    if not existing_pcs:
        print("  No structural PC columns found in data")
        return None

    dimorphism_results = analysis.analyze(feature_columns=existing_pcs)
    dimorphism_results.to_csv(dimorphism_output, index=False)
    print(f"Saved results to {dimorphism_output}")

    n_significant = dimorphism_results["Significant"].sum()
    print(f"Found {n_significant} significant features (FDR < 0.05)")

    # Show top 5
    print("\nTop 5 features by effect size:")
    top5 = dimorphism_results.head(5)[["Feature", "Cohen_D", "P_Adjusted"]]
    print(top5.to_string(index=False))

    return dimorphism_results


def run_ml_classification(data, output_dir: Path):
    """Run ML classification for gender prediction."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    from brain_connectome.models import ConnectomeRandomForest

    ml_output = output_dir / "ml_results.csv"

    # Prepare features
    struct_pcs = [c for c in data.columns if c.startswith("Struct_PC")]
    func_pcs = [c for c in data.columns if c.startswith("Func_PC")]
    feature_cols = struct_pcs + func_pcs

    if not feature_cols:
        print("  No PC columns found for ML classification")
        return None

    X = data[feature_cols].values
    y = (data["Gender"] == "M").astype(int).values

    # Remove any rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} subjects")
    print(f"Test set: {len(X_test)} subjects")

    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    clf = ConnectomeRandomForest(n_estimators=500, random_state=42)
    clf.fit(X_train, y_train, feature_names=feature_cols)

    # Evaluate
    metrics = clf.evaluate(X_test, y_test)
    print(f"Test Accuracy: {metrics['accuracy']:.2%}")

    # Save feature importance
    importance = clf.get_top_features(n=20)
    importance.to_csv(ml_output, index=False)
    print(f"Saved feature importance to {ml_output}")

    print("\nTop 10 features for gender classification:")
    print(importance.head(10).to_string(index=False))

    return importance, metrics


def generate_plots(data, output_dir: Path, dimorphism_results=None, ml_results=None):
    """Generate visualization plots."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd

    from brain_connectome.visualization import (
        plot_dimorphism_comparison,
        plot_feature_importance,
        plot_pca_scatter,
        plot_scree,
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    print(f"Saving plots to {plots_dir}/")

    # 1. PCA scatter plots
    struct_pc1 = "Struct_PC1" if "Struct_PC1" in data.columns else None
    struct_pc2 = "Struct_PC2" if "Struct_PC2" in data.columns else None

    if struct_pc1 and struct_pc2 and "Gender" in data.columns:
        print("  Generating PCA scatter plot...")
        fig, ax = plot_pca_scatter(
            data,
            pc_x=struct_pc1,
            pc_y=struct_pc2,
            hue="Gender",
            title="Structural Connectome: PC1 vs PC2 by Gender",
        )
        fig.savefig(plots_dir / "pca_scatter_struct.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    func_pc1 = "Func_PC1" if "Func_PC1" in data.columns else None
    func_pc2 = "Func_PC2" if "Func_PC2" in data.columns else None

    if func_pc1 and func_pc2 and "Gender" in data.columns:
        print("  Generating functional PCA scatter plot...")
        fig, ax = plot_pca_scatter(
            data,
            pc_x=func_pc1,
            pc_y=func_pc2,
            hue="Gender",
            title="Functional Connectome: PC1 vs PC2 by Gender",
        )
        fig.savefig(plots_dir / "pca_scatter_func.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2. Dimorphism plots
    if dimorphism_results is not None and len(dimorphism_results) > 0:
        # Top dimorphic feature comparison
        top_feature = dimorphism_results.iloc[0]["Feature"]
        if top_feature in data.columns:
            print(f"  Generating dimorphism plot for {top_feature}...")
            fig, ax = plot_dimorphism_comparison(
                data, feature=top_feature, title=f"Distribution of {top_feature} by Gender"
            )
            fig.savefig(plots_dir / "dimorphism_top_feature.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Cohen's D bar plot for top 20 features
        print("  Generating effect size bar plot...")
        top20 = dimorphism_results.head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#1f77b4" if d < 0 else "#d62728" for d in top20["Cohen_D"]]
        ax.barh(range(len(top20)), top20["Cohen_D"].values, color=colors)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["Feature"])
        ax.set_xlabel("Cohen's D (Effect Size)")
        ax.set_title("Sexual Dimorphism: Top 20 Features by Effect Size")
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(plots_dir / "dimorphism_effect_sizes.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. ML feature importance
    if ml_results is not None:
        importance_df, _ = ml_results
        print("  Generating feature importance plot...")
        fig, ax = plot_feature_importance(
            importance_df,
            n_features=15,
            title="Random Forest: Top 15 Features for Gender Classification",
        )
        fig.savefig(plots_dir / "ml_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. PCA variance plot (if available)
    variance_path = output_dir / "pca_variance.csv"
    if variance_path.exists():
        print("  Generating scree plot...")
        variance_df = pd.read_csv(variance_path)
        fig, ax = plot_scree(
            variance_df["Variance_Explained"].values,
            n_components=30,
            title="PCA Scree Plot: Variance Explained",
        )
        fig.savefig(plots_dir / "pca_scree.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5. VAE training history (if available)
    vae_history_path = output_dir / "vae_training_history.csv"
    if vae_history_path.exists():
        print("  Generating VAE training plot...")
        history = pd.read_csv(vae_history_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history["Epoch"], history["Train_Loss"], label="Training Loss")
        if history["Val_Loss"].notna().any():
            ax.plot(history["Epoch"], history["Val_Loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("VAE Training History")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(plots_dir / "vae_training.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Generated {len(list(plots_dir.glob('*.png')))} plots")


def run_pipeline(
    skip_pca: bool = False,
    skip_vae: bool = False,
    skip_dimorphism: bool = False,
    skip_ml: bool = False,
    skip_plots: bool = False,
) -> None:
    """Run the complete analysis pipeline."""
    import pandas as pd

    project_root = get_project_root()
    data_dir = project_root / "data"
    output_dir = project_root / "output"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Brain Connectome Analysis Pipeline")
    print("=" * 60)

    # Check for required data
    print("\nStep 0: Checking Data")
    print("-" * 40)
    if not check_data_exists(data_dir):
        print("\nERROR: Required data files not found.")
        print("Please download HCP data from ConnectomeDB:")
        print("  https://db.humanconnectome.org/")
        print("\nSee data/README.md for data organization instructions.")
        sys.exit(1)
    print("All required data files found.")

    has_raw_connectomes = check_raw_connectomes_exist(data_dir)
    if has_raw_connectomes:
        print("Raw connectome data available for PCA/VAE analysis.")
    else:
        print("Raw connectome data not found - will use pre-computed TNPCA coefficients.")

    # Step 1: Load Data
    print("\nStep 1: Loading Data")
    print("-" * 40)
    merged_data_path = data_dir / "processed" / "full_data.csv"

    if merged_data_path.exists():
        print(f"Loading cached merged dataset from {merged_data_path}")
        data = pd.read_csv(merged_data_path)
    else:
        from brain_connectome import ConnectomeDataLoader

        print("Loading and merging HCP data...")
        loader = ConnectomeDataLoader(str(data_dir))
        data = loader.load_merged_dataset()
        merged_data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(merged_data_path, index=False)
        print(f"Saved merged dataset to {merged_data_path}")

    print(f"Dataset loaded: {data.shape[0]} subjects, {data.shape[1]} features")

    # Step 2: PCA Analysis
    if not skip_pca and has_raw_connectomes:
        print("\nStep 2: PCA Analysis")
        print("-" * 40)
        pca_output = output_dir / "pca_variance.csv"
        if pca_output.exists():
            print(f"PCA results already exist at {pca_output}")
        else:
            run_pca_analysis(data_dir, output_dir)
    else:
        reason = "--skip-pca" if skip_pca else "no raw connectome data"
        print(f"\nStep 2: Skipping PCA Analysis ({reason})")

    # Step 3: VAE Analysis
    if not skip_vae and has_raw_connectomes:
        print("\nStep 3: VAE Analysis")
        print("-" * 40)
        vae_output = output_dir / "vae_latent.csv"
        if vae_output.exists():
            print(f"VAE results already exist at {vae_output}")
        else:
            run_vae_analysis(data_dir, output_dir, epochs=200)
    else:
        reason = "--skip-vae" if skip_vae else "no raw connectome data"
        print(f"\nStep 3: Skipping VAE Analysis ({reason})")

    # Step 4: Dimorphism Analysis
    dimorphism_results = None
    if not skip_dimorphism:
        print("\nStep 4: Sexual Dimorphism Analysis")
        print("-" * 40)
        dimorphism_output = output_dir / "dimorphism_results.csv"
        if dimorphism_output.exists():
            print(f"Dimorphism results already exist at {dimorphism_output}")
            dimorphism_results = pd.read_csv(dimorphism_output)
            n_significant = dimorphism_results["Significant"].sum()
            print(f"Found {n_significant} significant features (FDR < 0.05)")
        else:
            dimorphism_results = run_dimorphism_analysis(data, output_dir)
    else:
        print("\nStep 4: Skipping Dimorphism Analysis (--skip-dimorphism)")

    # Step 5: ML Classification
    ml_results = None
    if not skip_ml:
        print("\nStep 5: Machine Learning Classification")
        print("-" * 40)
        ml_output = output_dir / "ml_results.csv"
        if ml_output.exists():
            print(f"ML results already exist at {ml_output}")
            importance = pd.read_csv(ml_output)
            ml_results = (importance, {"accuracy": "cached"})
            print("\nTop 10 features for gender classification:")
            print(importance.head(10).to_string(index=False))
        else:
            ml_results = run_ml_classification(data, output_dir)
    else:
        print("\nStep 5: Skipping ML Classification (--skip-ml)")

    # Step 6: Visualization
    if not skip_plots:
        print("\nStep 6: Generating Visualizations")
        print("-" * 40)
        generate_plots(data, output_dir, dimorphism_results, ml_results)
    else:
        print("\nStep 6: Skipping Visualization (--skip-plots)")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - pca_variance.csv, pca_scores.csv (if PCA ran)")
    print("  - vae_latent.csv, vae_training_history.csv (if VAE ran)")
    print("  - dimorphism_results.csv (if dimorphism analysis ran)")
    print("  - ml_results.csv (if ML classification ran)")
    print("  - plots/ (if visualization ran)")


def main():
    """Run the analysis pipeline from command line."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run the Brain Connectome analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Runners/run_pipeline.py                    # Run full pipeline
  python Runners/run_pipeline.py --quick            # Quick mode (skip PCA/VAE/plots)
  python Runners/run_pipeline.py --skip-vae         # Skip just VAE
  python Runners/run_pipeline.py --skip-plots       # Skip visualization
        """,
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip the PCA analysis step",
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip the VAE analysis step",
    )
    parser.add_argument(
        "--skip-dimorphism",
        action="store_true",
        help="Skip the dimorphism analysis step",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip the ML classification step",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip the visualization step",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip PCA, VAE, and plots (useful for testing)",
    )
    args = parser.parse_args()

    # Quick mode sets multiple skip flags
    if args.quick:
        args.skip_pca = True
        args.skip_vae = True
        args.skip_plots = True

    # Setup environment
    print("Step 0: Environment Setup")
    print("-" * 40)
    python_path, in_venv = setup_environment()

    if not in_venv:
        relaunch_in_venv(python_path)

    # Run pipeline
    run_pipeline(
        skip_pca=args.skip_pca,
        skip_vae=args.skip_vae,
        skip_dimorphism=args.skip_dimorphism,
        skip_ml=args.skip_ml,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
