"""Unit tests for newly integrated preprocessing, baseline CNN, and evaluation modules."""

import numpy as np
import pytest
import torch

from src.preprocessing.artefact_removal import remove_hair_dullrazor, filter_artefacts
from src.preprocessing.illumination import normalize_illumination_gray_world, apply_clahe_lab
from src.features.feature_extractor import preprocess_and_extract_features
from src.diagnosis.baseline_cnn import BlackBoxResNetBaseline
from src.evaluation.evaluate_fuzzy_concepts import evaluate_fuzzy_concepts
from src.evaluation.evaluate_explainability import evaluate_explainability
from src.evaluation.benchmark_system import benchmark_pipeline


def test_hair_removal():
    img = np.full((128, 128, 3), 200, dtype=np.uint8)
    # Draw dark artificial hair line
    img[64, :] = 10
    cleaned = remove_hair_dullrazor(img, kernel_size=5, threshold=8)
    assert cleaned.shape == img.shape
    assert cleaned.dtype == np.uint8
    # Hair pixel value should be significantly restored towards background
    assert cleaned[64, 64, 0] > 100


def test_illumination_normalization():
    # Synthetic image with strong blue tint
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # R
    img[:, :, 1] = 50   # G
    img[:, :, 2] = 200  # B
    balanced = normalize_illumination_gray_world(img, p=6)
    assert balanced.shape == img.shape
    # Channel values should be pulled closer together
    diff_orig = abs(int(img[0, 0, 2]) - int(img[0, 0, 0]))
    diff_norm = abs(int(balanced[0, 0, 2]) - int(balanced[0, 0, 0]))
    assert diff_norm < diff_orig


def test_preprocess_and_extract_features():
    img = np.random.randint(50, 200, size=(128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[30:90, 30:90] = 1

    feats, cleaned = preprocess_and_extract_features(img, mask)
    assert isinstance(feats, dict)
    assert "asymmetry_index" in feats
    assert "border_irregularity" in feats
    assert "color_variation" in feats
    assert "normalized_diameter" in feats
    assert cleaned.shape == img.shape


def test_blackbox_baseline_cnn():
    model = BlackBoxResNetBaseline(num_classes=7)
    dummy_input = torch.randn(2, 3, 256, 256)
    logits, probs = model(dummy_input)
    assert logits.shape == (2, 7)
    assert probs.shape == (2, 7)
    assert torch.allclose(torch.sum(probs, dim=-1), torch.tensor([1.0, 1.0]), atol=1e-4)


def test_system_benchmark():
    res = benchmark_pipeline(num_iterations=2)
    assert "latencies" in res
    assert "throughput_fps" in res
    assert "hair_removal_ms" in res["latencies"]
