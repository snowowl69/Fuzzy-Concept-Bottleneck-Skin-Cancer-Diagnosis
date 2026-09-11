"""Tests for Counterfactual Explanation and Intervention Engine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.diagnosis.concept_bottleneck import FuzzyConceptBottleneckModel
from src.counterfactual.concept_intervention import ConceptInterventionEngine
from src.counterfactual.counterfactual_generator import CounterfactualGenerator


def test_intervention_engine():
    model = FuzzyConceptBottleneckModel()
    engine = ConceptInterventionEngine(model)
    orig_c = torch.tensor([[0.8, 0.8, 0.8, 0.8]], dtype=torch.float32)

    res = engine.intervene_and_predict(orig_c, interventions={0: 0.1, 1: 0.1})

    assert "original_diagnosis" in res
    assert "intervened_diagnosis" in res
    assert res["intervened_concepts"]["Asymmetry Score"] == pytest.approx(0.1, abs=1e-4)


def test_counterfactual_generator():
    model = FuzzyConceptBottleneckModel()
    cf_gen = CounterfactualGenerator(model)
    orig_c = torch.tensor([[0.85, 0.85, 0.85, 0.75]], dtype=torch.float32)

    res = cf_gen.generate_counterfactual_concept(orig_c, target_class_idx=0)

    assert "explanation_text" in res
    assert res["target_class_idx"] == 0
    assert len(res["counterfactual_concepts"]) == 4


def test_counterfactual_rendering():
    model = FuzzyConceptBottleneckModel()
    cf_gen = CounterfactualGenerator(model)

    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8)
    cf_concepts = {
        "Asymmetry Score": 0.2,
        "Border Irregularity Score": 0.2,
        "Color Variation Score": 0.2,
        "Diameter Score": 0.3,
    }

    cf_img, cf_mask = cf_gen.render_counterfactual_image(image, mask, cf_concepts)

    assert cf_img.shape == image.shape
    assert cf_mask.shape == mask.shape


if __name__ == "__main__":
    test_intervention_engine()
    test_counterfactual_generator()
    test_counterfactual_rendering()
    print("All counterfactual tests passed!")
