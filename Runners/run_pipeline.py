#!/usr/bin/env python3
"""
Brain Connectome Analysis Pipeline Runner

Platform-independent script to run the complete analysis pipeline.
Automatically sets up virtual environment and installs dependencies.

Usage:
    python Runners/run_pipeline.py [--skip-dimorphism] [--skip-ml]

The pipeline includes:
    1. Data Loading - Load and merge HCP data
    2. PCA Analysis - Dimensionality reduction on connectomes
    3. Dimorphism Analysis - Sexual dimorphism statistical tests
    4. ML Classification - Train gender classifier
"""

import argparse
import sys
from pathlib import Path

# Handle imports for both direct execution and module import
try:
    from Runners.setup_environment import relaunch_in_venv, setup_environment
except ImportError:
    from setup_environment import relaunch_in_venv, setup_environment


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


def run_pipeline(skip_dimorphism: bool = False, skip_ml: bool = False) -> None:
    """Run the complete analysis pipeline."""
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

    # Import after environment setup
    import numpy as np
    import pandas as pd

    from brain_connectome import ConnectomeDataLoader, DimorphismAnalysis
    from brain_connectome.models import ConnectomeRandomForest

    # Step 1: Load Data
    print("\nStep 1: Loading Data")
    print("-" * 40)
    merged_data_path = data_dir / "processed" / "full_data.csv"

    if merged_data_path.exists():
        print(f"Loading cached merged dataset from {merged_data_path}")
        data = pd.read_csv(merged_data_path)
    else:
        print("Loading and merging HCP data...")
        loader = ConnectomeDataLoader(str(data_dir))
        data = loader.load_merged_dataset()
        merged_data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(merged_data_path, index=False)
        print(f"Saved merged dataset to {merged_data_path}")

    print(f"Dataset loaded: {data.shape[0]} subjects, {data.shape[1]} features")

    # Step 2: Dimorphism Analysis
    if not skip_dimorphism:
        print("\nStep 2: Sexual Dimorphism Analysis")
        print("-" * 40)
        dimorphism_output = output_dir / "dimorphism_results.csv"

        if dimorphism_output.exists():
            print(f"Dimorphism results already exist at {dimorphism_output}")
            dimorphism_results = pd.read_csv(dimorphism_output)
        else:
            print("Running dimorphism analysis on structural PCs...")
            analysis = DimorphismAnalysis(data)
            struct_pcs = [f"Struct_PC{i}" for i in range(1, 61)]
            dimorphism_results = analysis.analyze(feature_columns=struct_pcs)
            dimorphism_results.to_csv(dimorphism_output, index=False)
            print(f"Saved results to {dimorphism_output}")

        n_significant = dimorphism_results["Significant"].sum()
        print(f"Found {n_significant} significant features (FDR < 0.05)")

        # Show top 5
        print("\nTop 5 features by effect size:")
        top5 = dimorphism_results.head(5)[["Feature", "Cohen_D", "P_Adjusted"]]
        print(top5.to_string(index=False))
    else:
        print("\nStep 2: Skipping Dimorphism Analysis (--skip-dimorphism)")

    # Step 3: ML Classification
    if not skip_ml:
        print("\nStep 3: Machine Learning Classification")
        print("-" * 40)
        ml_output = output_dir / "ml_results.csv"

        # Prepare features
        struct_pcs = [c for c in data.columns if c.startswith("Struct_PC")]
        func_pcs = [c for c in data.columns if c.startswith("Func_PC")]
        feature_cols = struct_pcs + func_pcs

        X = data[feature_cols].values
        y = (data["Gender"] == "M").astype(int).values

        # Remove any rows with NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # Train/test split
        from sklearn.model_selection import train_test_split

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
    else:
        print("\nStep 3: Skipping ML Classification (--skip-ml)")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - dimorphism_results.csv (if ran)")
    print("  - ml_results.csv (if ran)")


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the Brain Connectome analysis pipeline")
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
    args = parser.parse_args()

    # Setup environment
    print("Step 0: Environment Setup")
    print("-" * 40)
    python_path, in_venv = setup_environment()

    if not in_venv:
        relaunch_in_venv(python_path)

    # Run pipeline
    run_pipeline(
        skip_dimorphism=args.skip_dimorphism,
        skip_ml=args.skip_ml,
    )


if __name__ == "__main__":
    main()
