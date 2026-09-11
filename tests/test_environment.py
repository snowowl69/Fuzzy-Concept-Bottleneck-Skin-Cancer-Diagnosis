"""Environment and project configuration verification test."""

from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_project_structure():
    root = Path(__file__).resolve().parents[1]
    required_dirs = [
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
    for d in required_dirs:
        path = root / d
        assert path.is_dir(), f"Missing directory: {path}"


def test_config_file():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "config.json"
    assert config_path.is_file()

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    assert config.get("project_name") == "Fuzzy Concept-Bottleneck Skin Cancer Diagnosis"
