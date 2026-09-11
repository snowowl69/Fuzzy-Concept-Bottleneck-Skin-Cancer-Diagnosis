"""Fuzzy Concept Bottleneck Model.

Combines geometric feature extraction, adaptive fuzzy concept generation, and diagnosis head
into a single intervenable end-to-end architecture.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.features.feature_extractor import extract_geometric_features
from src.fuzzy.fuzzy_concept_generator import AdaptiveFuzzyConceptGenerator
from src.diagnosis.diagnosis_head import DiagnosisHead, DIAGNOSIS_CLASSES


class FuzzyConceptBottleneckModel(nn.Module):
    """Fuzzy Concept Bottleneck Model.

    Pipeline:
    Image + Mask -> Geometric Features -> Adaptive Fuzzy Concept Generator -> Concept Vector C -> Diagnosis Head -> Prediction
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.concept_generator = AdaptiveFuzzyConceptGenerator()
        self.diagnosis_head = DiagnosisHead(num_concepts=4, num_classes=num_classes)

    def forward(
        self,
        features_or_concepts: torch.Tensor,
        intervention: Optional[Dict[int, float]] = None,
        is_concept_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through concept bottleneck.

        Args:
            features_or_concepts: Feature tensor (B, 4) or Concept tensor (B, 4).
            intervention: Dict mapping concept_index -> intervened_value (float in [0, 1]).
            is_concept_input: If True, input is already concept tensor.

        Returns:
            Tuple of (concepts, logits, probabilities)
        """
        if is_concept_input:
            concepts = features_or_concepts.clone()
        else:
            concepts = self.concept_generator(features_or_concepts)

        # Apply intervention if specified
        if intervention is not None:
            concepts = concepts.clone()
            for idx, val in intervention.items():
                concepts[:, idx] = val

        logits, probabilities = self.diagnosis_head(concepts)
        return concepts, logits, probabilities

    def predict_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        intervention: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Union[Dict, str, float, np.ndarray]]:
        """End-to-end prediction from raw numpy image and binary mask.

        Args:
            image: RGB image numpy array (H, W, 3)
            mask: Binary mask numpy array (H, W)
            intervention: Optional dict mapping concept_index (0..3) to target membership [0, 1].

        Returns:
            Dictionary with extracted features, fuzzy concept scores, intervention status, and diagnosis results.
        """
        # 1. Extract geometric features
        features = extract_geometric_features(image, mask)

        # 2. Convert to tensor
        feat_tensor = torch.tensor(
            [[
                features["asymmetry_index"],
                features["border_irregularity"],
                features["color_variation"],
                features["normalized_diameter"],
            ]],
            dtype=torch.float32,
        )

        # 3. Forward pass
        self.eval()
        with torch.no_grad():
            concepts, logits, probs = self.forward(feat_tensor, intervention=intervention)

        concept_names = self.concept_generator.get_concept_names()
        concept_dict = {
            concept_names[i]: float(concepts[0, i].item()) for i in range(4)
        }

        pred_idx = torch.argmax(probs, dim=-1).item()
        prob_dict = {
            DIAGNOSIS_CLASSES.get(i, f"Class_{i}"): float(probs[0, i].item())
            for i in range(probs.shape[1])
        }

        return {
            "extracted_features": features,
            "fuzzy_concepts": concept_dict,
            "concept_vector": concepts.squeeze(0).numpy(),
            "intervention_applied": intervention is not None,
            "interventions": intervention,
            "predicted_class_id": pred_idx,
            "predicted_class_name": DIAGNOSIS_CLASSES.get(pred_idx, "Unknown"),
            "class_probabilities": prob_dict,
            "confidence": float(probs[0, pred_idx].item()),
        }
