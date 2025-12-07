"""Command-line interface for connectopy package.

This module provides the entry point for the `connectopy` CLI command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Run the connectopy CLI."""
    parser = argparse.ArgumentParser(
        prog="connectopy",
        description="Brain Connectome Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  pipeline     Run the full analysis pipeline
  mediation    Run mediation analysis on HCP data
  alcohol      Run alcohol classification analysis

Examples:
  connectopy pipeline --quick
  connectopy mediation
  connectopy alcohol --variants tnpca --model-types rf
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline subcommand
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full analysis pipeline")
    pipeline_parser.add_argument("--skip-pca", action="store_true", help="Skip PCA analysis")
    pipeline_parser.add_argument("--skip-vae", action="store_true", help="Skip VAE analysis")
    pipeline_parser.add_argument(
        "--skip-dimorphism", action="store_true", help="Skip dimorphism analysis"
    )
    pipeline_parser.add_argument("--skip-ml", action="store_true", help="Skip ML classification")
    pipeline_parser.add_argument(
        "--skip-mediation", action="store_true", help="Skip mediation analysis"
    )
    pipeline_parser.add_argument("--skip-plots", action="store_true", help="Skip visualization")
    pipeline_parser.add_argument(
        "--quick", action="store_true", help="Quick mode (skip PCA, VAE, and plots)"
    )

    # Mediation subcommand
    subparsers.add_parser("mediation", help="Run mediation analysis on HCP data")

    # Alcohol subcommand
    alcohol_parser = subparsers.add_parser("alcohol", help="Run alcohol classification analysis")
    alcohol_parser.add_argument("--data-path", type=str, help="Path to full_data.csv")
    alcohol_parser.add_argument("--output-dir", type=str, help="Output directory")
    alcohol_parser.add_argument(
        "--variants",
        type=str,
        default="tnpca,vae,pca",
        help="Comma-separated list of variants",
    )
    alcohol_parser.add_argument(
        "--model-types",
        type=str,
        default="rf,ebm",
        help="Comma-separated model types",
    )

    # Version
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Find project root (parent of connectopy package)
    project_root = Path(__file__).parent.parent

    if args.command == "pipeline":
        # Import and run pipeline
        sys.path.insert(0, str(project_root))
        try:
            from Runners.run_pipeline import run_pipeline

            if args.quick:
                args.skip_pca = True
                args.skip_vae = True
                args.skip_plots = True

            run_pipeline(
                skip_pca=args.skip_pca,
                skip_vae=args.skip_vae,
                skip_dimorphism=args.skip_dimorphism,
                skip_ml=args.skip_ml,
                skip_mediation=args.skip_mediation,
                skip_plots=args.skip_plots,
            )
        except ImportError as e:
            print(f"Error importing pipeline: {e}", file=sys.stderr)
            print("Make sure all dependencies are installed.", file=sys.stderr)
            return 1

    elif args.command == "mediation":
        sys.path.insert(0, str(project_root))
        try:
            from Runners.run_mediation_hcp import main as run_mediation

            run_mediation()
        except ImportError as e:
            print(f"Error importing mediation runner: {e}", file=sys.stderr)
            return 1

    elif args.command == "alcohol":
        sys.path.insert(0, str(project_root))
        try:
            from Runners.run_alcohol_analysis import run_alcohol_analysis

            data_path = (
                Path(args.data_path)
                if args.data_path
                else project_root / "data" / "processed" / "full_data.csv"
            )
            output_dir = (
                Path(args.output_dir)
                if args.output_dir
                else project_root / "output" / "alcohol_analysis"
            )
            variants = [v.strip() for v in args.variants.split(",")]
            model_types = [m.strip() for m in args.model_types.split(",")]

            output_dir.mkdir(parents=True, exist_ok=True)

            run_alcohol_analysis(
                data_path=data_path,
                output_dir=output_dir,
                variants=variants,
                model_types=model_types,
            )
        except ImportError as e:
            print(f"Error importing alcohol analysis runner: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
