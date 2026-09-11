"""Geometric and Color Feature Extraction Module.

Extracts ABCDE-inspired visual measurements from skin lesion images and binary segmentation masks:
- Lesion Area
- Lesion Perimeter
- Lesion Diameter (equivalent & maximum)
- Aspect Ratio
- Circularity / Compactness
- Border Irregularity Index
- Image Moment-based Asymmetry Index
- Color Channel Statistics (RGB/HSV standard deviations & variation)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


def extract_geometric_features(
    image: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Extract geometric and color features from an image and binary lesion mask.

    Args:
        image: RGB image as numpy array (H, W, 3) in range [0, 255] or [0, 1].
        mask: Binary mask as numpy array (H, W) where 1/255 represents lesion.

    Returns:
        Dictionary containing extracted raw and normalized feature measurements.
    """
    # Ensure binary mask (0 or 1)
    mask_binary = ((mask > 0.5) | (mask > 127)).astype(np.uint8)
    if mask.ndim == 3:
        mask_binary = mask_binary[:, :, 0]

    # Handle RGB image format
    image_np = np.asarray(image, dtype=np.float32)
    if image_np.max() <= 1.0:
        image_np = image_np * 255.0

    H, W = mask_binary.shape
    total_image_pixels = float(H * W)

    # 1. Area
    area = float(np.sum(mask_binary))

    if area < 1.0:
        # Fallback for empty mask
        return {
            "area": 0.0,
            "normalized_area": 0.0,
            "perimeter": 0.0,
            "normalized_perimeter": 0.0,
            "equivalent_diameter": 0.0,
            "max_diameter": 0.0,
            "normalized_diameter": 0.0,
            "aspect_ratio": 1.0,
            "circularity": 1.0,
            "border_irregularity": 0.0,
            "asymmetry_index": 0.0,
            "color_std_r": 0.0,
            "color_std_g": 0.0,
            "color_std_b": 0.0,
            "color_variation": 0.0,
        }

    normalized_area = float(area / total_image_pixels)

    # 2. Perimeter (count boundary pixels using 4-connectivity)
    padded_mask = np.pad(mask_binary, 1, mode="constant", constant_values=0)
    boundary = (
        (padded_mask[1:-1, 1:-1] == 1)
        & (
            (padded_mask[:-2, 1:-1] == 0)
            | (padded_mask[2:, 1:-1] == 0)
            | (padded_mask[1:-1, :-2] == 0)
            | (padded_mask[1:-1, 2:] == 0)
        )
    )
    perimeter = float(np.sum(boundary))
    normalized_perimeter = float(perimeter / (2.0 * (H + W)))

    # 3. Diameter
    equivalent_diameter = float(2.0 * np.sqrt(area / np.pi))
    max_diameter = float(np.sqrt(H**2 + W**2))
    
    # Bounding box dimensions
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_h = float(rmax - rmin + 1)
    bbox_w = float(cmax - cmin + 1)
    bounding_diameter = float(np.sqrt(bbox_h**2 + bbox_w**2))
    normalized_diameter = float(equivalent_diameter / np.sqrt(total_image_pixels))

    # 4. Aspect Ratio
    major_axis = max(bbox_h, bbox_w)
    minor_axis = max(min(bbox_h, bbox_w), 1.0)
    aspect_ratio = float(major_axis / minor_axis)

    # 5. Circularity / Compactness: 4 * pi * area / (perimeter^2)
    circularity = float(min(1.0, (4.0 * np.pi * area) / (perimeter**2 + 1e-8)))

    # 6. Border Irregularity Index: ratio of perimeter to perimeter of circle with same area
    circle_perimeter = 2.0 * np.sqrt(np.pi * area)
    border_irregularity = float(max(0.0, (perimeter - circle_perimeter) / (circle_perimeter + 1e-8)))

    # 7. Asymmetry Index (Image moment / Flipped mask overlap)
    # Crop to bounding box
    cropped_mask = mask_binary[rmin : rmax + 1, cmin : cmax + 1]
    # Horizontal flip asymmetry
    h_flip = np.fliplr(cropped_mask)
    h_diff = np.sum(np.abs(cropped_mask - h_flip)) / (area + 1e-8)
    # Vertical flip asymmetry
    v_flip = np.flipud(cropped_mask)
    v_diff = np.sum(np.abs(cropped_mask - v_flip)) / (area + 1e-8)
    asymmetry_index = float(min(1.0, (h_diff + v_diff) / 2.0))

    # 8. Color Statistics inside lesion
    lesion_pixels = image_np[mask_binary == 1]  # shape (N, 3)
    if len(lesion_pixels) > 0:
        std_r = float(np.std(lesion_pixels[:, 0]))
        std_g = float(np.std(lesion_pixels[:, 1]))
        std_b = float(np.std(lesion_pixels[:, 2]))
        color_std_mean = float((std_r + std_g + std_b) / 3.0)
        # Bounded color variation score in [0, 1] assuming max std ~ 64
        color_variation = float(min(1.0, color_std_mean / 64.0))
    else:
        std_r, std_g, std_b, color_variation = 0.0, 0.0, 0.0, 0.0

    return {
        "area": area,
        "normalized_area": normalized_area,
        "perimeter": perimeter,
        "normalized_perimeter": normalized_perimeter,
        "equivalent_diameter": equivalent_diameter,
        "bounding_diameter": bounding_diameter,
        "normalized_diameter": normalized_diameter,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "border_irregularity": border_irregularity,
        "asymmetry_index": asymmetry_index,
        "color_std_r": std_r,
        "color_std_g": std_g,
        "color_std_b": std_b,
        "color_variation": color_variation,
    }


def preprocess_and_extract_features(
    image: np.ndarray,
    mask: np.ndarray,
    apply_hair_removal: bool = True,
    apply_illumination_norm: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Preprocess image with artefact cleaning and extract geometric/color features.

    Returns:
        Tuple of (features_dict, preprocessed_image)
    """
    cleaned_img = image.copy()
    if apply_hair_removal:
        try:
            from src.preprocessing.artefact_removal import remove_hair_dullrazor
            cleaned_img = remove_hair_dullrazor(cleaned_img)
        except Exception:
            pass

    if apply_illumination_norm:
        try:
            from src.preprocessing.illumination import normalize_illumination_gray_world
            cleaned_img = normalize_illumination_gray_world(cleaned_img)
        except Exception:
            pass

    features = extract_geometric_features(cleaned_img, mask)
    return features, cleaned_img


def compute_fourier_descriptors(
    mask: np.ndarray,
    num_descriptors: int = 8,
) -> np.ndarray:
    """Compute Fourier shape descriptors from the lesion boundary contour.

    Uses the radial distance from the centroid to each contour point,
    then computes the FFT and returns normalized magnitude coefficients.

    Args:
        mask: Binary mask (H, W) with lesion = 1.
        num_descriptors: Number of Fourier coefficients to return.

    Returns:
        Array of shape (num_descriptors,) with normalized Fourier descriptor magnitudes.
    """
    import cv2

    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros(num_descriptors, dtype=np.float64)

    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.squeeze()

    if contour_pts.ndim != 2 or len(contour_pts) < 4:
        return np.zeros(num_descriptors, dtype=np.float64)

    # Compute centroid
    cx = np.mean(contour_pts[:, 0])
    cy = np.mean(contour_pts[:, 1])

    # Radial distance from centroid to each contour point
    radii = np.sqrt((contour_pts[:, 0] - cx) ** 2 + (contour_pts[:, 1] - cy) ** 2)

    # Compute FFT of radial signal
    fft_coeffs = np.fft.fft(radii)
    magnitudes = np.abs(fft_coeffs)

    # Normalize by DC component (index 0) to make scale-invariant
    dc = magnitudes[0] if magnitudes[0] > 1e-8 else 1.0
    normalized = magnitudes[1:num_descriptors + 1] / dc

    # Pad if contour has fewer points than requested descriptors
    if len(normalized) < num_descriptors:
        normalized = np.pad(normalized, (0, num_descriptors - len(normalized)))

    return normalized[:num_descriptors]


def normalize_features(
    features: Dict[str, float],
    feature_ranges: Dict[str, Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Min-max normalize all feature values to [0, 1].

    Args:
        features: Raw feature dictionary.
        feature_ranges: Optional dict of {feature_name: (min_val, max_val)}.
            If not provided, uses sensible defaults based on typical dermoscopy images.

    Returns:
        Dictionary with all values normalized to [0, 1].
    """
    defaults: Dict[str, Tuple[float, float]] = {
        "area": (0.0, 65536.0),
        "normalized_area": (0.0, 1.0),
        "perimeter": (0.0, 2048.0),
        "normalized_perimeter": (0.0, 1.0),
        "equivalent_diameter": (0.0, 512.0),
        "bounding_diameter": (0.0, 512.0),
        "normalized_diameter": (0.0, 1.0),
        "aspect_ratio": (1.0, 10.0),
        "circularity": (0.0, 1.0),
        "border_irregularity": (0.0, 3.0),
        "asymmetry_index": (0.0, 1.0),
        "color_std_r": (0.0, 128.0),
        "color_std_g": (0.0, 128.0),
        "color_std_b": (0.0, 128.0),
        "color_variation": (0.0, 1.0),
    }

    ranges = feature_ranges or defaults
    normalized = {}

    for key, val in features.items():
        if key.startswith("fourier_"):
            # Fourier descriptors are already scale-normalized
            normalized[key] = float(np.clip(val, 0.0, 1.0))
            continue

        lo, hi = ranges.get(key, (0.0, 1.0))
        if hi - lo < 1e-8:
            normalized[key] = 0.0
        else:
            normalized[key] = float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

    return normalized


def validate_features(
    features: Dict[str, float],
) -> Tuple[bool, Dict[str, str]]:
    """Validate extracted features for NaN, Inf, and out-of-range values.

    Args:
        features: Feature dictionary to validate.

    Returns:
        Tuple of (all_valid, issues_dict). issues_dict maps feature names to error messages.
    """
    issues: Dict[str, str] = {}

    if not features:
        return False, {"general": "Feature dictionary is empty."}

    for key, val in features.items():
        if not isinstance(val, (int, float, np.integer, np.floating)):
            issues[key] = f"Non-numeric value: {type(val).__name__}"
            continue

        fval = float(val)

        if np.isnan(fval):
            issues[key] = "Value is NaN"
        elif np.isinf(fval):
            issues[key] = "Value is Inf"
        elif fval < -1e6 or fval > 1e6:
            issues[key] = f"Extreme value: {fval}"

    # Check critical features exist
    required_keys = {"asymmetry_index", "border_irregularity", "color_variation", "normalized_diameter"}
    missing = required_keys - set(features.keys())
    if missing:
        issues["missing_keys"] = f"Missing required features: {sorted(missing)}"

    all_valid = len(issues) == 0
    return all_valid, issues


