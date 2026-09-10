"""Test basic image processing with a synthetic lesion image.

This is an engineering test only.
It does not use real medical data or perform diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    """Create a synthetic image, segment it, and extract features."""

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Create a synthetic image
    # ---------------------------------------------------------
    image = np.full((256, 256, 3), 240, dtype=np.uint8)

    # Draw a simple dark oval to represent a synthetic lesion.
    cv2.ellipse(
        image,
        center=(128, 128),
        axes=(55, 75),
        angle=20,
        startAngle=0,
        endAngle=360,
        color=(80, 50, 40),
        thickness=-1,
    )

    # ---------------------------------------------------------
    # 2. Create a simple segmentation mask
    # ---------------------------------------------------------
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the dark lesion from the light background.
    _, mask = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # ---------------------------------------------------------
    # 3. Extract basic geometric features
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError("No lesion contour was found.")

    # Select the largest contour.
    largest_contour = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, True))

    if perimeter == 0:
        raise RuntimeError("Contour perimeter is zero.")

    # Compactness is a simple shape descriptor.
    compactness = float(
        (4 * np.pi * area) / (perimeter ** 2)
    )

    # ---------------------------------------------------------
    # 4. Save outputs
    # ---------------------------------------------------------
    image_path = output_dir / "synthetic_lesion.png"
    mask_path = output_dir / "synthetic_lesion_mask.png"
    feature_path = output_dir / "synthetic_lesion_features.json"

    # OpenCV uses BGR; Pillow expects RGB.
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(
        image_path
    )

    Image.fromarray(mask).save(mask_path)

    features = {
        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "compactness": compactness,
    }

    with feature_path.open("w", encoding="utf-8") as file:
        json.dump(features, file, indent=2)

    # ---------------------------------------------------------
    # 5. Verify outputs
    # ---------------------------------------------------------
    if not image_path.is_file():
        raise RuntimeError("Synthetic image was not saved.")

    if not mask_path.is_file():
        raise RuntimeError("Synthetic mask was not saved.")

    if not feature_path.is_file():
        raise RuntimeError("Feature file was not saved.")

    print("=== Image Processing Test ===")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Lesion area: {area:.2f} pixels")
    print(f"Lesion perimeter: {perimeter:.2f} pixels")
    print(f"Compactness: {compactness:.4f}")
    print(f"Saved image: {image_path}")
    print(f"Saved mask: {mask_path}")
    print(f"Saved features: {feature_path}")
    print("=== Image processing test successful ===")


if __name__ == "__main__":
    main()