"""Verify the initial development environment.

This script checks that:
- Python is running.
- The project root exists.
- The initial project directories exist.
- The configuration file can be loaded.
- The virtual environment is being used.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


def main() -> None:
    """Run the Phase 0 environment checks."""

    # Project root:
    # verify_environment.py
    # -> utils
    # -> src
    # -> project root
    project_root = Path(__file__).resolve().parents[2]

    required_directories = [
        "src",
        "src/config",
        "src/data",
        "src/segmentation",
        "src/features",
        "src/fuzzy",
        "src/diagnosis",
        "src/counterfactual",
        "src/utils",
        "tests",
        "configs",
        "data/raw",
        "data/processed",
        "data/masks",
        "models",
        "notebooks",
        "reports",
        "logs",
    ]

    print("=== Phase 0: Environment Verification ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Project root: {project_root}")

    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")

    if not project_root.exists():
        raise RuntimeError("Project root does not exist.")

    print("\nChecking project directories...")

    for directory in required_directories:
        path = project_root / directory

        if not path.is_dir():
            raise RuntimeError(f"Missing directory: {path}")

        print(f"[OK] {directory}")

    config_path = project_root / "configs" / "config.json"

    print("\nChecking configuration file...")

    if not config_path.is_file():
        raise RuntimeError(f"Missing configuration file: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("project_name") != (
        "Fuzzy Concept-Bottleneck Skin Cancer Diagnosis"
    ):
        raise RuntimeError("Unexpected project name in config.json.")

    print("[OK] config.json loaded successfully")

    virtual_environment = project_root / ".venv"

    print("\nChecking virtual environment...")

    if virtual_environment.is_dir():
        print("[OK] .venv directory exists")
    else:
        raise RuntimeError("The .venv directory was not found.")

    print("\n=== Verification successful ===")
    print("Phase 0 initial project structure is ready.")


if __name__ == "__main__":
    main()