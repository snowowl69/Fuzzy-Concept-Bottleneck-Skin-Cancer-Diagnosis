"""Tests for Adaptive Fuzzy Concept Generator."""

from __future__ import annotations

import torch
from src.fuzzy.fuzzy_concept_generator import AdaptiveFuzzyConceptGenerator


def test_fuzzy_generator_forward():
    generator = AdaptiveFuzzyConceptGenerator()
    features = torch.tensor([[0.2, 0.1, 0.4, 0.5], [0.8, 0.9, 0.7, 0.8]], dtype=torch.float32)
    concepts = generator(features)

    assert concepts.shape == (2, 4)
    assert (concepts >= 0.0).all() and (concepts <= 1.0).all()


def test_fuzzy_generator_dict_input():
    generator = AdaptiveFuzzyConceptGenerator()
    feat_dict = {
        "asymmetry_index": 0.5,
        "border_irregularity": 0.3,
        "color_variation": 0.6,
        "normalized_diameter": 0.4,
    }
    concepts = generator(feat_dict)
    assert concepts.shape == (1, 4)
    assert 0.0 <= concepts[0, 0].item() <= 1.0


def test_gradient_backprop():
    generator = AdaptiveFuzzyConceptGenerator()
    features = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    concepts = generator(features)
    loss = concepts.sum()
    loss.backward()

    assert features.grad is not None
    assert generator.asymmetry_mf.slope.grad is not None


if __name__ == "__main__":
    test_fuzzy_generator_forward()
    test_fuzzy_generator_dict_input()
    test_gradient_backprop()
    print("All fuzzy generator tests passed!")
