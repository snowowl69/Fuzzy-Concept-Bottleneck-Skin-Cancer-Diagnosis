"""Fuzzy Concept Intervention Module.

Allows clinical or analytical overriding of continuous fuzzy concept values to observe
diagnosis sensitivity and measure feature influence on model predictions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch

from src.diagnosis.concept_bottleneck import FuzzyConceptBottleneckModel
from src.diagnosis.diagnosis_head import DIAGNOSIS_CLASSES


class ConceptInterventionEngine:
    """Intervention Engine for Concept-Level Sensitivity Analysis."""

    def __init__(self, model: FuzzyConceptBottleneckModel):
        self.model = model
        self.concept_names = [
            "Asymmetry Score",
            "Border Irregularity Score",
            "Color Variation Score",
            "Diameter Score",
        ]

    def intervene_and_predict(
        self,
        original_concepts: torch.Tensor,
        interventions: Dict[int, float],
    ) -> Dict:
        """Apply targeted intervention values and evaluate recomputed diagnosis.

        Args:
            original_concepts: Tensor of shape (1, 4) or (4,)
            interventions: Dict mapping concept_index (0..3) -> target_value [0, 1]

        Returns:
            Dict containing original and updated predictions and confidence changes.
        """
        self.model.eval()
        if original_concepts.dim() == 1:
            original_concepts = original_concepts.unsqueeze(0)

        with torch.no_grad():
            orig_concepts, orig_logits, orig_probs = self.model(
                original_concepts, is_concept_input=True
            )
            intervened_concepts, new_logits, new_probs = self.model(
                original_concepts, intervention=interventions, is_concept_input=True
            )

        orig_pred = torch.argmax(orig_probs, dim=-1).item()
        new_pred = torch.argmax(new_probs, dim=-1).item()

        return {
            "original_concepts": {
                self.concept_names[i]: float(orig_concepts[0, i].item()) for i in range(4)
            },
            "intervened_concepts": {
                self.concept_names[i]: float(intervened_concepts[0, i].item()) for i in range(4)
            },
            "interventions": interventions,
            "original_diagnosis": DIAGNOSIS_CLASSES.get(orig_pred, "Unknown"),
            "intervened_diagnosis": DIAGNOSIS_CLASSES.get(new_pred, "Unknown"),
            "diagnosis_flipped": orig_pred != new_pred,
            "original_probabilities": {
                DIAGNOSIS_CLASSES.get(i, f"Class_{i}"): float(orig_probs[0, i].item())
                for i in range(orig_probs.shape[1])
            },
            "intervened_probabilities": {
                DIAGNOSIS_CLASSES.get(i, f"Class_{i}"): float(new_probs[0, i].item())
                for i in range(new_probs.shape[1])
            },
        }

    def compute_concept_sensitivity(
        self,
        base_concepts: torch.Tensor,
        target_class_idx: int = 1,  # e.g. Melanoma
    ) -> Dict[str, float]:
        """Compute marginal prediction sensitivity for each fuzzy concept."""
        sensitivity = {}
        for idx, name in enumerate(self.concept_names):
            res_min = self.intervene_and_predict(base_concepts, {idx: 0.0})
            res_max = self.intervene_and_predict(base_concepts, {idx: 1.0})

            p_min = res_min["intervened_probabilities"].get(
                DIAGNOSIS_CLASSES.get(target_class_idx, "Melanoma"), 0.0
            )
            p_max = res_max["intervened_probabilities"].get(
                DIAGNOSIS_CLASSES.get(target_class_idx, "Melanoma"), 0.0
            )

            sensitivity[name] = float(p_max - p_min)

        return sensitivity
