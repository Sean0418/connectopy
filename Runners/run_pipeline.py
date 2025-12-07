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
    --skip-mediation     Skip mediation analysis
    --skip-plots         Skip visualization generation
    --quick              Run quick mode (skip PCA, VAE, and plots)

The pipeline includes:
    1. Data Loading - Load and merge HCP data
    2. PCA Analysis - Dimensionality reduction on raw connectomes
    3. VAE Analysis - Variational autoencoder for nonlinear embeddings
    4. Dimorphism Analysis - Sexual dimorphism statistical tests
    5. ML Classification - Alcohol use disorder prediction (RF + EBM, sex-stratified)
    6. Mediation Analysis - Brain network mediation of cognitive-alcohol relationships
    7. Visualization - Generate analysis plots
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

    from connectopy.analysis import ConnectomePCA

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

    from connectopy.analysis import ConnectomeVAE

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
    from connectopy import DimorphismAnalysis

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
    """Run ML classification for alcohol use disorder prediction.

    Uses Random Forest and EBM with:
    - Cognitive + connectome features
    - Sex stratification (separate models for M/F)
    - fit_with_cv() for GridSearchCV and class imbalance handling
    - Comprehensive metrics (AUC, balanced accuracy, etc.)
    """
    import numpy as np
    import pandas as pd

    from connectopy.models import (
        ConnectomeEBM,
        ConnectomeRandomForest,
        get_cognitive_features,
        get_connectome_features,
    )

    # Create alcohol target from SSAGA_Alc_D4_Ab_Dx if not present
    if "alc_y" not in data.columns:
        if "SSAGA_Alc_D4_Ab_Dx" in data.columns:
            # HCP coding: 1 = No diagnosis, 5 = Yes diagnosis
            data = data.copy()
            data["alc_y"] = np.where(data["SSAGA_Alc_D4_Ab_Dx"] == 5, 1, 0).astype(int)
            print("Created alcohol target from SSAGA_Alc_D4_Ab_Dx")
        else:
            print("  ERROR: No alcohol target column found (need 'alc_y' or 'SSAGA_Alc_D4_Ab_Dx')")
            return None

    # Check for Gender column
    if "Gender" not in data.columns:
        print("  ERROR: 'Gender' column not found for sex stratification")
        return None

    # Get feature sets
    cog_features = get_cognitive_features(data, include_age=True)
    conn_features = get_connectome_features(data, "tnpca")

    # Combine cognitive + connectome features
    feature_cols = cog_features + conn_features
    feature_cols = [c for c in feature_cols if c in data.columns]

    if not feature_cols:
        print("  No features found for ML classification")
        return None

    n_cog = len(cog_features)
    n_conn = len(conn_features)
    print(f"Features: {n_cog} cognitive + {n_conn} connectome = {len(feature_cols)} total")

    # Show target distribution
    print("\nAlcohol target distribution:")
    print(f"  0 (no diagnosis): {(data['alc_y'] == 0).sum()}")
    print(f"  1 (alcohol abuse/dependence): {(data['alc_y'] == 1).sum()}")

    all_results = []

    # Train models stratified by sex
    for sex in ["M", "F"]:
        df_sex = data[data["Gender"] == sex].copy()
        if df_sex.empty or len(df_sex) < 50:
            print(f"\n  [WARN] Insufficient data for sex={sex}, skipping")
            continue

        # Prepare data
        cols = feature_cols + ["alc_y"]
        sub = df_sex[cols].dropna()

        if len(sub) < 50:
            print(f"\n  [WARN] Insufficient non-NA data for sex={sex}, skipping")
            continue

        X = sub[feature_cols].values
        y = sub["alc_y"].astype(int).values

        if len(np.unique(y)) < 2:
            print(f"\n  [WARN] Only one class for sex={sex}, skipping")
            continue

        pos_rate = y.mean()
        print(f"\n{'='*50}")
        print(f"Training models for Sex={sex}")
        print(f"{'='*50}")
        print(f"  Samples: {len(y)}, Positive rate: {pos_rate:.3f}")

        # Train Random Forest with CV
        print("\n  Training Random Forest with GridSearchCV...")
        rf = ConnectomeRandomForest(n_estimators=500, class_weight="balanced", random_state=42)
        try:
            rf_metrics = rf.fit_with_cv(
                X,
                y,
                feature_names=feature_cols,
                handle_imbalance=True,
                param_grid={
                    "rf__n_estimators": [300, 500],
                    "rf__max_depth": [None, 10],
                    "rf__min_samples_leaf": [1, 5],
                },
            )
            rf_metrics["sex"] = sex
            rf_metrics["model"] = "RF"
            all_results.append(rf_metrics)
            print(f"    CV AUC: {rf_metrics['cv_best_auc']:.3f}")
            print(f"    Test AUC: {rf_metrics['test_auc']:.3f}")
            print(f"    Test Balanced Accuracy: {rf_metrics['test_bal_acc']:.3f}")

            # Save RF feature importance
            rf_imp = rf.get_top_features(n=20)
            rf_imp.to_csv(output_dir / f"rf_importance_{sex}.csv", index=False)

        except Exception as e:
            print(f"    [ERROR] RF training failed: {e}")

        # Train EBM with CV
        print("\n  Training EBM with GridSearchCV...")
        try:
            ebm = ConnectomeEBM(
                max_bins=32,
                learning_rate=0.01,
                max_leaves=3,
                interactions=0,
                random_state=42,
            )
            ebm_metrics = ebm.fit_with_cv(
                X,
                y,
                feature_names=feature_cols,
                handle_imbalance=True,
                param_grid={
                    "max_leaves": [2, 3],
                    "min_samples_leaf": [20, 50],
                },
            )
            ebm_metrics["sex"] = sex
            ebm_metrics["model"] = "EBM"
            all_results.append(ebm_metrics)
            print(f"    CV AUC: {ebm_metrics['cv_best_auc']:.3f}")
            print(f"    Test AUC: {ebm_metrics['test_auc']:.3f}")
            print(f"    Test Balanced Accuracy: {ebm_metrics['test_bal_acc']:.3f}")

            # Save EBM feature importance
            ebm_imp = ebm.get_top_features(n=20)
            ebm_imp.to_csv(output_dir / f"ebm_importance_{sex}.csv", index=False)

        except ImportError:
            print("    [WARN] interpret package not installed, skipping EBM")
        except Exception as e:
            print(f"    [ERROR] EBM training failed: {e}")

    # Save summary results
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Reorder columns
        first_cols = [
            "model",
            "sex",
            "n_train",
            "n_test",
            "cv_best_auc",
            "test_auc",
            "test_bal_acc",
        ]
        other_cols = [c for c in results_df.columns if c not in first_cols]
        col_order = [c for c in first_cols if c in results_df.columns] + other_cols
        results_df = results_df[col_order]  # type: ignore[assignment]

        ml_output = output_dir / "alcohol_classification_results.csv"
        results_df.to_csv(ml_output, index=False)
        print(f"\nSaved classification results to {ml_output}")

        # Print summary
        print("\n" + "=" * 60)
        print("ALCOHOL CLASSIFICATION SUMMARY")
        print("=" * 60)
        summary = results_df[["model", "sex", "cv_best_auc", "test_auc", "test_bal_acc"]]
        print(summary.to_string(index=False))

        return results_df, all_results
    else:
        print("\n  No models were successfully trained.")
        return None


def run_mediation_analysis(data_dir: Path, output_dir: Path) -> "dict | None":
    """Run sex-stratified mediation analysis.

    Tests whether brain networks (SC/FC) mediate the relationship between
    cognitive traits and alcohol dependence, stratified by sex.

    Model: Cognitive → Brain Network → Alcohol Dependence
    """
    from connectopy.analysis import run_multiple_mediations
    from connectopy.data import (
        create_alcohol_severity_score,
        create_composite_scores,
        load_merged_hcp_data,
    )

    print("Loading HCP data for mediation analysis...")

    try:
        # Load and merge all data
        merged = load_merged_hcp_data(
            data_dir / "raw",
            n_sc_components=10,
            n_fc_components=10,
        )

        # Create composite scores
        merged = create_composite_scores(merged)
        merged = create_alcohol_severity_score(merged)

        print(f"  Loaded {len(merged)} subjects with all required data")

    except (FileNotFoundError, ImportError) as e:
        print(f"  Could not load HCP data: {e}")
        print("  Skipping mediation analysis.")
        return None

    # Define variables for mediation
    cognitive_cols = ["FluidComposite", "CrystalComposite", "ListSort_Unadj", "PMAT24_A_CR"]
    brain_cols = [f"SC_PC{i}" for i in range(1, 6)] + [f"FC_PC{i}" for i in range(1, 6)]
    alcohol_col = "AlcoholSeverity"

    # Filter to available columns
    cognitive_cols = [c for c in cognitive_cols if c in merged.columns]
    brain_cols = [c for c in brain_cols if c in merged.columns]

    if not cognitive_cols:
        print("  No cognitive columns available")
        return None

    if not brain_cols:
        print("  No brain network columns available")
        return None

    if alcohol_col not in merged.columns:
        print(f"  Alcohol column '{alcohol_col}' not found")
        return None

    # Remove rows with missing values
    analysis_cols = cognitive_cols + brain_cols + [alcohol_col, "Gender"]
    merged_clean = merged.dropna(subset=analysis_cols)
    print(f"  {len(merged_clean)} subjects after removing missing values")

    if len(merged_clean) < 100:
        print("  Insufficient subjects for mediation analysis")
        return None

    # Run mediation analyses
    print(f"\nTesting {len(cognitive_cols)} cognitive × {len(brain_cols)} brain pathways...")
    results = run_multiple_mediations(
        data=merged_clean,
        cognitive_cols=cognitive_cols,
        brain_cols=brain_cols,
        alcohol_col=alcohol_col,
        sex_col="Gender",
        n_bootstrap=1000,
        random_state=42,
    )

    # Save results
    mediation_output = output_dir / "mediation_results.csv"
    results.to_csv(mediation_output, index=False)
    print(f"\nSaved results to {mediation_output}")

    # Summary
    n_male_sig = results["male_significant"].sum()
    n_female_sig = results["female_significant"].sum()
    n_sex_diff = results["sex_diff_significant"].sum()

    print("\n" + "=" * 50)
    print("MEDIATION ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total pathways tested: {len(results)}")
    print(f"Significant in males: {n_male_sig}")
    print(f"Significant in females: {n_female_sig}")
    print(f"Significant sex differences: {n_sex_diff}")

    # Show top sex differences
    if n_sex_diff > 0:
        print("\nTop pathways with sex differences:")
        sex_diff = results[results["sex_diff_significant"]].copy()
        sex_diff["abs_diff"] = sex_diff["sex_difference"].abs()
        top_diff = sex_diff.nlargest(5, "abs_diff")
        for _, row in top_diff.iterrows():
            print(f"  {row['cognitive']} → {row['brain_network']}:")
            print(f"    Male: {row['male_indirect']:.4f}, Female: {row['female_indirect']:.4f}")
            print(f"    Difference: {row['sex_difference']:.4f}")

    return results.to_dict()


def generate_plots(data, output_dir: Path, dimorphism_results=None):
    """Generate visualization plots."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd

    from connectopy.visualization import (
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

    # 3. ML feature importance (from alcohol classification)
    # Check for sex-stratified importance files
    for sex in ["M", "F"]:
        rf_imp_path = output_dir / f"rf_importance_{sex}.csv"
        if rf_imp_path.exists():
            print(f"  Generating RF feature importance plot for {sex}...")
            importance_df = pd.read_csv(rf_imp_path)
            fig, ax = plot_feature_importance(
                importance_df,
                n_features=15,
                title=f"Random Forest: Top 15 Features ({sex}) - Alcohol Classification",
            )
            fig.savefig(
                plots_dir / f"rf_feature_importance_{sex}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        ebm_imp_path = output_dir / f"ebm_importance_{sex}.csv"
        if ebm_imp_path.exists():
            print(f"  Generating EBM feature importance plot for {sex}...")
            importance_df = pd.read_csv(ebm_imp_path)
            fig, ax = plot_feature_importance(
                importance_df,
                n_features=15,
                title=f"EBM: Top 15 Features ({sex}) - Alcohol Classification",
            )
            fig.savefig(
                plots_dir / f"ebm_feature_importance_{sex}.png", dpi=150, bbox_inches="tight"
            )
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
    skip_mediation: bool = False,
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
        from connectopy import ConnectomeDataLoader

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

    # Step 5: ML Classification (Alcohol Use Disorder)
    if not skip_ml:
        print("\nStep 5: Alcohol Use Disorder Classification")
        print("-" * 40)
        ml_output = output_dir / "alcohol_classification_results.csv"
        if ml_output.exists():
            print(f"Classification results already exist at {ml_output}")
            results_df = pd.read_csv(ml_output)
            print("\nSummary of existing results:")
            summary_cols = ["model", "sex", "cv_best_auc", "test_auc", "test_bal_acc"]
            summary_cols = [c for c in summary_cols if c in results_df.columns]
            print(results_df[summary_cols].to_string(index=False))
        else:
            run_ml_classification(data, output_dir)
    else:
        print("\nStep 5: Skipping ML Classification (--skip-ml)")

    # Step 6: Mediation Analysis
    if not skip_mediation:
        print("\nStep 6: Mediation Analysis")
        print("-" * 40)
        mediation_output = output_dir / "mediation_results.csv"
        if mediation_output.exists():
            print(f"Mediation results already exist at {mediation_output}")
            mediation_df = pd.read_csv(mediation_output)
            n_sex_diff = mediation_df["sex_diff_significant"].sum()
            print(f"Found {n_sex_diff} pathways with significant sex differences")
        else:
            run_mediation_analysis(data_dir, output_dir)
    else:
        print("\nStep 6: Skipping Mediation Analysis (--skip-mediation)")

    # Step 7: Visualization
    if not skip_plots:
        print("\nStep 7: Generating Visualizations")
        print("-" * 40)
        generate_plots(data, output_dir, dimorphism_results)
    else:
        print("\nStep 7: Skipping Visualization (--skip-plots)")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - pca_variance.csv, pca_scores.csv (if PCA ran)")
    print("  - vae_latent.csv, vae_training_history.csv (if VAE ran)")
    print("  - dimorphism_results.csv (if dimorphism analysis ran)")
    print("  - alcohol_classification_results.csv (if ML ran)")
    print("  - rf_importance_M.csv, rf_importance_F.csv (RF feature importance by sex)")
    print("  - ebm_importance_M.csv, ebm_importance_F.csv (EBM feature importance by sex)")
    print("  - mediation_results.csv (if mediation analysis ran)")
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
        "--skip-mediation",
        action="store_true",
        help="Skip the mediation analysis step",
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
        skip_mediation=args.skip_mediation,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
