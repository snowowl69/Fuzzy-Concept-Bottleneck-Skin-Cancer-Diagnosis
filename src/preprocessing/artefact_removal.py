"""Hair and Artefact Removal for Dermoscopic Images (DullRazor Implementation).

Uses morphological Black Top-Hat filtering to identify dark, thin hair structures
and inpainting (Telea or Navier-Stokes) to remove them without altering lesion morphology.
"""

from __future__ import annotations

import cv2
import numpy as np


def remove_hair_dullrazor(
    image: np.ndarray,
    kernel_size: int = 9,
    threshold: int = 10,
    inpaint_radius: int = 3,
    method: int = cv2.INPAINT_TELEA,
) -> np.ndarray:
    """Apply DullRazor hair removal algorithm to a dermoscopic RGB image.

    Args:
        image: RGB image as uint8 numpy array of shape (H, W, 3).
        kernel_size: Size of the structuring element for morphological filtering.
        threshold: Intensity threshold for hair mask extraction.
        inpaint_radius: Neighborhood diameter for inpainting around hair pixels.
        method: OpenCV inpaint method (cv2.INPAINT_TELEA or cv2.INPAINT_NS).

    Returns:
        Dermoscopic RGB image with hair removed (H, W, 3) uint8.
    """
    if image is None or image.size == 0:
        return image

    # 1. Convert to grayscale for morphological operations
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Black Top-Hat transform (Closing - Original Image) to isolate dark thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Threshold to generate binary hair mask
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # 4. Refine hair mask using morphological dilation to cover hair borders
    refine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.dilate(hair_mask, refine_kernel, iterations=1)

    # 5. Inpaint hair regions using surrounding healthy dermis/lesion context
    inpainted = cv2.inpaint(image, hair_mask, inpaint_radius, method)

    return inpainted


def filter_artefacts(
    image: np.ndarray,
    remove_vignetting: bool = True,
    remove_hair: bool = True,
) -> np.ndarray:
    """General dermoscopy artefact cleaning pipeline.

    Args:
        image: RGB image array.
        remove_vignetting: If True, crops or softens dark circular dermatoscope borders.
        remove_hair: If True, applies DullRazor hair removal.

    Returns:
        Cleaned RGB image.
    """
    cleaned = image.copy()

    if remove_hair:
        cleaned = remove_hair_dullrazor(cleaned)

    if remove_vignetting:
        # Detect dark corner vignetting and soften via circular mask blending
        h, w = cleaned.shape[:2]
        center_y, center_x = h // 2, w // 2
        max_r = min(center_y, center_x) * 0.98

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        vignette_mask = dist_from_center > max_r

        if np.any(vignette_mask):
            mean_border_color = np.median(cleaned[~vignette_mask], axis=(0,))
            cleaned[vignette_mask] = mean_border_color.astype(np.uint8)

    return cleaned
