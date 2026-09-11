"""Adaptive Fuzzy Concept Generator.

Maps extracted raw geometric and color features into continuous fuzzy clinical concepts:
- Asymmetry
- Border Irregularity
- Color Variation
- Lesion Diameter

Uses trainable ANFIS-inspired membership functions for continuous concept vector C in [0, 1]^4.
"""

from __future__ import annotations

from typing import Dict, List, Union
import torch
import torch.nn as nn

from src.fuzzy.membership_functions import GaussianMembership, SigmoidMembership, LinearFuzzyMapping


class AdaptiveFuzzyConceptGenerator(nn.Module):
    """Adaptive Fuzzy Concept Generator Module.

    Takes 4-dimensional feature vector [Asymmetry, Irregularity, Color, Diameter]
    and computes fuzzy concept membership scores C in [0, 1]^4.
    """

    def __init__(self):
        super().__init__()
        # Trainable membership parameters initialized to reasonable clinical defaults
        self.asymmetry_mf = SigmoidMembership(init_slope=8.0, init_center=0.3)
        self.irregularity_mf = SigmoidMembership(init_slope=10.0, init_center=0.25)
        self.color_mf = SigmoidMembership(init_slope=12.0, init_center=0.3)
        self.diameter_mf = SigmoidMembership(init_slope=8.0, init_center=0.4)

    def forward(self, features: Union[torch.Tensor, Dict[str, float]]) -> torch.Tensor:
        """Compute continuous fuzzy concept memberships.

        Args:
            features: Tensor of shape (B, 4) or dictionary of extracted feature floats.

        Returns:
            Tensor of shape (B, 4) containing continuous concepts:
            [c_asymmetry, c_border_irregularity, c_color_variation, c_diameter]
        """
        if isinstance(features, dict):
            # Convert dictionary to tensor (Batch size = 1)
            f_asym = torch.tensor([[features.get("asymmetry_index", 0.0)]], dtype=torch.float32)
            f_irreg = torch.tensor([[features.get("border_irregularity", 0.0)]], dtype=torch.float32)
            f_color = torch.tensor([[features.get("color_variation", 0.0)]], dtype=torch.float32)
            f_diam = torch.tensor([[features.get("normalized_diameter", 0.0)]], dtype=torch.float32)
            feat_tensor = torch.cat([f_asym, f_irreg, f_color, f_diam], dim=1)
        else:
            feat_tensor = features

        f_asym = feat_tensor[:, 0:1]
        f_irreg = feat_tensor[:, 1:2]
        f_color = feat_tensor[:, 2:3]
        f_diam = feat_tensor[:, 3:4]

        c_asym = self.asymmetry_mf(f_asym)
        c_irreg = self.irregularity_mf(f_irreg)
        c_color = self.color_mf(f_color)
        c_diam = self.diameter_mf(f_diam)

        concepts = torch.cat([c_asym, c_irreg, c_color, c_diam], dim=1)
        return torch.clamp(concepts, min=0.0, max=1.0)

    def get_concept_names(self) -> List[str]:
        return [
            "Asymmetry Score",
            "Border Irregularity Score",
            "Color Variation Score",
            "Diameter Score",
        ]

    def compute_concept_supervision_loss(
        self,
        predicted_concepts: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute concept supervision loss between predicted fuzzy concepts and proxy ground-truth.

        The proxy ground-truth is derived by clamping raw feature values to [0, 1],
        treating them as target concept memberships. This enforces consistency between
        the adaptive fuzzy concepts and the underlying geometric measurements.

        Args:
            predicted_concepts: Predicted fuzzy concept tensor of shape (B, 4).
            raw_features: Raw feature tensor of shape (B, 4) with values
                [asymmetry_index, border_irregularity, color_variation, normalized_diameter].

        Returns:
            Scalar MSE supervision loss.
        """
        # Derive proxy concept targets from raw features (clamped to [0, 1])
        concept_targets = torch.clamp(raw_features, 0.0, 1.0)
        return torch.nn.functional.mse_loss(predicted_concepts, concept_targets)

    def get_membership_parameters(self) -> Dict[str, Dict[str, float]]:
        """Expose current learned membership function parameters for inspection/logging.

        Returns:
            Dictionary mapping concept name to its membership function parameters.
        """
        params = {}
        mf_map = {
            "Asymmetry": self.asymmetry_mf,
            "Border_Irregularity": self.irregularity_mf,
            "Color_Variation": self.color_mf,
            "Diameter": self.diameter_mf,
        }
        for name, mf in mf_map.items():
            mf_params = {}
            for pname, pval in mf.named_parameters():
                mf_params[pname] = float(pval.item())
            params[name] = mf_params

        return params

