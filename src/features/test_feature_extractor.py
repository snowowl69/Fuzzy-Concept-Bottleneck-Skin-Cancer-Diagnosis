"""Tests for geometric and color feature extraction."""

from __future__ import annotations

import numpy as np
from src.features.feature_extractor import extract_geometric_features


def test_circle_features():
    """Test feature extraction on a clean circle image/mask."""
    size = 100
    y, x = np.ogrid[:size, :size]
    center = (50, 50)
    radius = 20
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2).astype(np.uint8)

    # Constant color image
    image = np.ones((size, size, 3), dtype=np.uint8) * 128

    features = extract_geometric_features(image, mask)

    assert features["area"] > 0
    assert features["equivalent_diameter"] > 0
    assert 0.0 <= features["circularity"] <= 1.0
    assert 0.0 <= features["asymmetry_index"] <= 1.0
    assert features["color_variation"] == 0.0  # constant color -> 0 std


def test_empty_mask_features():
    """Test feature extraction with an empty binary mask."""
    size = 50
    mask = np.zeros((size, size), dtype=np.uint8)
    image = np.zeros((size, size, 3), dtype=np.uint8)

    features = extract_geometric_features(image, mask)

    assert features["area"] == 0.0
    assert features["circularity"] == 1.0
    assert features["asymmetry_index"] == 0.0


def test_color_variation():
    """Test color variation extraction on non-uniform image."""
    size = 50
    mask = np.ones((size, size), dtype=np.uint8)
    image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

    features = extract_geometric_features(image, mask)

    assert features["color_variation"] > 0.0


if __name__ == "__main__":
    test_circle_features()
    test_empty_mask_features()
    test_color_variation()
    print("All feature extraction tests passed!")
