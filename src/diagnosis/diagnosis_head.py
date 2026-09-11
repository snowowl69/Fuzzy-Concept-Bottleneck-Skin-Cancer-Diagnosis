"""Diagnosis Head Module.

Lightweight, interpretable classifier mapping continuous fuzzy concepts vector C to diagnostic probabilities.
Restricted to input C in [0, 1]^4 to ensure concept bottleneck interpretability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


DIAGNOSIS_CLASSES = {
    0: "Melanoma",
    1: "Melanocytic Nevus",
    2: "Basal Cell Carcinoma",
    3: "Actinic Keratosis",
    4: "Benign Keratosis",
    5: "Dermatofibroma",
    6: "Vascular Lesion",
}


class DiagnosisHead(nn.Module):
    """Interpretable Diagnosis Classifier operating strictly on fuzzy concepts."""

    def __init__(self, num_concepts: int = 4, num_classes: int = 7, hidden_dim: int = 16):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes),
        )

        # Direct linear layer for baseline feature attribution analysis
        self.linear_attribution = nn.Linear(num_concepts, num_classes)

    def forward(self, concepts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            concepts: Tensor of shape (B, num_concepts) in range [0, 1].

        Returns:
            Tuple of (logits, probabilities) both of shape (B, num_classes).
        """
        if concepts.dim() == 1:
            concepts = concepts.unsqueeze(0)

        logits = self.classifier(concepts)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities

    def predict(self, concepts: torch.Tensor) -> Dict[str, Any]:
        """Predict class label and return probability breakdown."""
        self.eval()
        with torch.no_grad():
            logits, probs = self.forward(concepts)
            pred_idx = torch.argmax(probs, dim=-1).item()
            prob_dict = {
                DIAGNOSIS_CLASSES.get(i, f"Class_{i}"): float(probs[0, i].item())
                for i in range(self.num_classes)
            }
            return {
                "predicted_class_id": pred_idx,
                "predicted_class_name": DIAGNOSIS_CLASSES.get(pred_idx, "Unknown"),
                "class_probabilities": prob_dict,
                "confidence": float(probs[0, pred_idx].item()),
            }

    def compute_concept_attribution(self, concepts: torch.Tensor = None) -> Dict[str, Any]:
        """Compute per-concept weight contributions to each diagnosis class.

        Uses the linear_attribution layer to produce interpretable
        concept-to-diagnosis traceability weights.

        Args:
            concepts: Optional concept tensor (1, num_concepts). If provided,
                also computes the per-concept contribution (weight * concept_value)
                for the specific input.

        Returns:
            Dictionary containing:
                - 'weight_matrix': Raw weight matrix (num_classes, num_concepts)
                - 'bias_vector': Bias terms per class
                - 'concept_names': List of concept labels
                - 'class_names': List of class labels
                - 'per_concept_contribution': (only if concepts provided) weighted contribution per concept per class
                - 'most_influential_concept': (only if concepts provided) concept with highest absolute weight per class
        """
        with torch.no_grad():
            weight = self.linear_attribution.weight.detach().cpu().numpy()  # (num_classes, num_concepts)
            bias = self.linear_attribution.bias.detach().cpu().numpy()      # (num_classes,)

        concept_names = [
            "Asymmetry", "Border Irregularity", "Color Variation", "Diameter"
        ]
        class_names = [
            DIAGNOSIS_CLASSES.get(i, f"Class_{i}") for i in range(self.num_classes)
        ]

        result: Dict[str, Any] = {
            "weight_matrix": {class_names[c]: {concept_names[j]: float(weight[c, j])
                              for j in range(min(self.num_concepts, len(concept_names)))}
                              for c in range(self.num_classes)},
            "bias_vector": {class_names[c]: float(bias[c]) for c in range(self.num_classes)},
            "concept_names": concept_names[:self.num_concepts],
            "class_names": class_names,
        }

        if concepts is not None:
            c_vals = concepts.detach().cpu().numpy()
            if c_vals.ndim == 1:
                c_vals = c_vals.reshape(1, -1)
            # Contribution = weight * concept_value for each (class, concept)
            contributions = weight * c_vals  # broadcasting: (num_classes, num_concepts)
            result["per_concept_contribution"] = {
                class_names[c]: {concept_names[j]: float(contributions[c, j])
                                 for j in range(min(self.num_concepts, len(concept_names)))}
                for c in range(self.num_classes)
            }
            # Most influential concept per class (by absolute contribution)
            most_influential = {}
            for c in range(self.num_classes):
                abs_contribs = [abs(contributions[c, j]) for j in range(min(self.num_concepts, len(concept_names)))]
                best_idx = int(max(range(len(abs_contribs)), key=lambda i: abs_contribs[i]))
                most_influential[class_names[c]] = concept_names[best_idx]
            result["most_influential_concept"] = most_influential

        return result
