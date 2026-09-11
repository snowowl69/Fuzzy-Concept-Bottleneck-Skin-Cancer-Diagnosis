"""Illumination Normalization and Color Constancy for Dermoscopic Images.

Implements:
1. Shades-of-Gray / Gray-World color constancy algorithm (Minkowski p-norm).
2. Contrast-Limited Adaptive Histogram Equalization (CLAHE) in CIELAB color space.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_illumination_gray_world(
    image: np.ndarray,
    p: int = 6,
) -> np.ndarray:
    """Apply Shades-of-Gray / Gray-World color constancy to correct dermatoscope lighting bias.

    Args:
        image: RGB image uint8 (H, W, 3).
        p: Minkowski p-norm. p=1 is Gray-World, p=6 is Shades-of-Gray, p=inf is Max-RGB.

    Returns:
        Color-balanced RGB image (H, W, 3) uint8.
    """
    if image is None or image.size == 0:
        return image

    img_float = image.astype(np.float64)

    # Compute Minkowski norm for each channel (R, G, B)
    if p == 1:
        # Standard Gray-World: average channel intensity
        illuminant = np.mean(img_float, axis=(0, 1))
    elif p == float("inf"):
        # Max-RGB algorithm
        illuminant = np.max(img_float, axis=(0, 1))
    else:
        # Shades-of-Gray (Minkowski p-norm)
        illuminant = np.power(np.mean(np.power(img_float, p), axis=(0, 1)), 1.0 / p)

    # Avoid zero division
    illuminant = np.maximum(illuminant, 1e-6)

    # Normalize illuminant so mean response across channels is preserved
    target_gray = np.mean(illuminant)
    gain = target_gray / illuminant

    # Scale channels
    normalized = img_float * gain
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return normalized


def apply_clahe_lab(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the L-channel of CIELAB.

    Preserves chromaticity (A and B channels) while enhancing local lesion contrast.

    Args:
        image: RGB image uint8 (H, W, 3).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced RGB image (H, W, 3) uint8.
    """
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return rgb_enhanced


def full_dermoscopy_preprocessing(
    image: np.ndarray,
    apply_hair_removal: bool = True,
    apply_color_constancy: bool = True,
    apply_clahe: bool = False,
) -> np.ndarray:
    """End-to-end dermoscopic image preprocessing pipeline.

    Combines artefact removal, color constancy, and optional contrast enhancement.
    """
    processed = image.copy()

    if apply_hair_removal:
        from src.preprocessing.artefact_removal import remove_hair_dullrazor
        processed = remove_hair_dullrazor(processed)

    if apply_color_constancy:
        processed = normalize_illumination_gray_world(processed, p=6)

    if apply_clahe:
        processed = apply_clahe_lab(processed, clip_limit=1.5)

    return processed
