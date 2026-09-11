"""Centralized Project Configuration Loader.

Loads project-wide settings from configs/config.json and provides typed access
to paths, settings, diagnosis class mappings, and device configuration.
Uses a singleton pattern so all modules share one configuration instance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.json"


class ProjectConfig:
    """Singleton project configuration loaded from configs/config.json."""

    _instance: Optional["ProjectConfig"] = None

    def __new__(cls) -> "ProjectConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self) -> None:
        if self._loaded:
            return
        self._load_config()
        self._loaded = True

    def _load_config(self) -> None:
        """Load configuration from JSON file, falling back to defaults if missing."""
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = {}

        # -- Paths --
        paths = raw.get("paths", {})
        self.raw_data_dir = _PROJECT_ROOT / paths.get("raw_data", "data/raw")
        self.processed_data_dir = _PROJECT_ROOT / paths.get("processed_data", "data/processed")
        self.masks_dir = _PROJECT_ROOT / paths.get("masks", "data/masks")
        self.models_dir = _PROJECT_ROOT / paths.get("models", "models")
        self.reports_dir = _PROJECT_ROOT / paths.get("reports", "reports")
        self.logs_dir = _PROJECT_ROOT / paths.get("logs", "logs")

        # -- Settings --
        settings = raw.get("settings", {})
        self.random_seed: int = int(settings.get("random_seed", 42))
        self.image_size: int = int(settings.get("image_size", 256))
        self.num_classes: int = int(settings.get("num_classes", 7))
        self.device_name: str = settings.get("device", "cpu")

        # -- Diagnosis Classes --
        classes_raw = raw.get("diagnosis_classes", {})
        self.diagnosis_classes: Dict[int, str] = {
            int(k): v for k, v in classes_raw.items()
        } if classes_raw else {
            0: "Melanoma",
            1: "Melanocytic Nevus",
            2: "Basal Cell Carcinoma",
            3: "Actinic Keratosis",
            4: "Benign Keratosis",
            5: "Dermatofibroma",
            6: "Vascular Lesion",
        }

    # -- Computed Properties --

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_name)

    @property
    def features_csv(self) -> Path:
        return self.processed_data_dir / "features.csv"

    @property
    def model_checkpoint_path(self) -> Path:
        return self.models_dir / "concept_bottleneck.pth"

    @property
    def segmentation_checkpoint_path(self) -> Path:
        return self.models_dir / "segmentation_unet_baseline.pt"

    def ensure_directories(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [self.models_dir, self.reports_dir, self.logs_dir,
                  self.processed_data_dir, self.raw_data_dir, self.masks_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def class_name(self, idx: int) -> str:
        """Get diagnosis class name by index."""
        return self.diagnosis_classes.get(idx, f"Class_{idx}")

    def __repr__(self) -> str:
        return (
            f"ProjectConfig(root={self.project_root}, "
            f"image_size={self.image_size}, "
            f"num_classes={self.num_classes}, "
            f"device={self.device_name})"
        )


def get_config() -> ProjectConfig:
    """Get the singleton ProjectConfig instance."""
    return ProjectConfig()
