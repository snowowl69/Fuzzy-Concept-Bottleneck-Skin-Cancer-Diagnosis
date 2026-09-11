"""Full End-to-End Pipeline Integration Test."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from src.features.feature_extractor import extract_geometric_features
from src.fuzzy.fuzzy_concept_generator import AdaptiveFuzzyConceptGenerator
from src.diagnosis.concept_bottleneck import FuzzyConceptBottleneckModel
from src.counterfactual.concept_intervention import ConceptInterventionEngine
from src.counterfactual.counterfactual_generator import CounterfactualGenerator


def test_full_pipeline():
    # 1. Generate synthetic image & lesion mask
    size = 128
    y, x = np.ogrid[:size, :size]
    center = (64, 64)
    radius = 30
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2).astype(np.uint8)

    image = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
    image[mask == 1, 0] = 220  # Make lesion region reddish

    # 2. End-to-End Model Prediction
    model = FuzzyConceptBottleneckModel(num_classes=7)
    pred_res = model.predict_image(image, mask)

    assert "extracted_features" in pred_res
    assert "fuzzy_concepts" in pred_res
    assert "predicted_class_name" in pred_res
    assert "class_probabilities" in pred_res

    # 3. Concept Intervention
    engine = ConceptInterventionEngine(model)
    orig_c = torch.tensor(np.array([pred_res["concept_vector"]]), dtype=torch.float32)
    interv_res = engine.intervene_and_predict(orig_c, interventions={0: 0.1, 1: 0.1})

    assert "intervened_diagnosis" in interv_res

    # 4. Counterfactual Generation
    cf_gen = CounterfactualGenerator(model)
    cf_res = cf_gen.generate_counterfactual_concept(orig_c, target_class_idx=0)

    assert "explanation_text" in cf_res

    # 5. Counterfactual Image Rendering
    cf_img, cf_mask = cf_gen.render_counterfactual_image(
        image, mask, cf_res["counterfactual_concepts"]
    )
    assert cf_img.shape == image.shape
    assert cf_mask.shape == mask.shape


if __name__ == "__main__":
    test_full_pipeline()
    print("Full pipeline test passed successfully!")
