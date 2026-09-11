"""Preprocessing package for dermoscopic image artefact removal and illumination normalization."""

from src.preprocessing.artefact_removal import remove_hair_dullrazor, filter_artefacts
from src.preprocessing.illumination import normalize_illumination_gray_world, apply_clahe_lab
from src.preprocessing.image_preprocessor import ImagePreprocessor

__all__ = [
    "remove_hair_dullrazor",
    "filter_artefacts",
    "normalize_illumination_gray_world",
    "apply_clahe_lab",
    "ImagePreprocessor",
]
