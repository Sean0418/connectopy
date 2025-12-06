"""
Environment Setup Module

Platform-independent module to create and manage Python virtual environments.
Automatically installs dependencies from requirements.txt.
"""

import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_venv_python() -> Path:
    """Get the path to the Python executable in the virtual environment."""
    project_root = get_project_root()
    venv_dir = project_root / ".venv"

    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def is_in_venv() -> bool:
    """Check if currently running inside a virtual environment."""
    return sys.prefix != sys.base_prefix


def create_venv() -> Path:
    """Create a virtual environment if it doesn't exist."""
    project_root = get_project_root()
    venv_dir = project_root / ".venv"
    python_path = get_venv_python()

    if not venv_dir.exists():
        print(f"Creating virtual environment at {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        print("Virtual environment created.")
    else:
        print(f"Virtual environment already exists at {venv_dir}")

    return python_path


def install_dependencies(python_path: Path) -> None:
    """Install dependencies from requirements.txt."""
    project_root = get_project_root()
    requirements_file = project_root / "requirements.txt"

    if not requirements_file.exists():
        print("Warning: requirements.txt not found, skipping dependency installation.")
        return

    print("Installing dependencies from requirements.txt...")
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "-r", str(requirements_file)],
        check=True,
    )

    # Also install the package in editable mode
    print("Installing brain-connectome package in editable mode...")
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "-e", str(project_root)],
        check=True,
    )

    print("Dependencies installed successfully.")


def setup_environment() -> tuple[Path, bool]:
    """
    Set up the Python environment.

    Returns
    -------
    python_path : Path
        Path to the Python executable in the virtual environment.
    in_venv : bool
        Whether currently running inside the virtual environment.
    """
    python_path = create_venv()
    in_venv = is_in_venv()

    if in_venv:
        print("Already running in virtual environment.")
    else:
        print("Not running in virtual environment.")
        install_dependencies(python_path)

    return python_path, in_venv


def relaunch_in_venv(python_path: Path) -> None:
    """Re-launch the current script inside the virtual environment."""
    print(f"\nRe-launching in virtual environment: {python_path}")
    print("-" * 40)

    # Get the original script that was run
    script = sys.argv[0]
    args = sys.argv[1:]

    # Re-run with venv Python
    result = subprocess.run([str(python_path), script] + args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    python_path, in_venv = setup_environment()
    print(f"\nPython path: {python_path}")
    print(f"In venv: {in_venv}")
