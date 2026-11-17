"""Counterfactual Explanation Generator.

Generates sparse, minimal concept modifications to flip diagnosis predictions
and renders segmentation-constrained visual counterfactual image representations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.diagnosis.concept_bottleneck import FuzzyConceptBottleneckModel
from src.diagnosis.diagnosis_head import DIAGNOSIS_CLASSES


class CounterfactualGenerator:
    """Generates concept-level counterfactual explanations and visual lesion renderings."""

    def __init__(self, model: FuzzyConceptBottleneckModel):
        self.model = model
        self.concept_names = [
            "Asymmetry Score",
            "Border Irregularity Score",
            "Color Variation Score",
            "Diameter Score",
        ]

    def generate_counterfactual_concept(
        self,
        original_concepts: torch.Tensor,
        target_class_idx: int = 0,
        steps: int = 150,
        lr: float = 0.08,
        l1_reg: float = 0.05,
    ) -> Dict:
        """Optimize concept shift delta to achieve target diagnosis with minimal change."""
        self.model.eval()
        if original_concepts.dim() == 1:
            original_concepts = original_concepts.unsqueeze(0)

        orig_c = original_concepts.detach().clone()
        with torch.no_grad():
            _, orig_probs = self.model.diagnosis_head(orig_c)
        orig_pred_idx = torch.argmax(orig_probs, dim=-1).item()
        orig_name = DIAGNOSIS_CLASSES.get(orig_pred_idx, "Unknown")
        target_name = DIAGNOSIS_CLASSES.get(target_class_idx, "Unknown")

        # Case 1: Model already predicts the target diagnosis
        if orig_pred_idx == target_class_idx:
            return {
                "target_class_idx": target_class_idx,
                "target_class_name": target_name,
                "success": True,
                "original_concepts": {
                    self.concept_names[i]: float(orig_c[0, i].item()) for i in range(4)
                },
                "counterfactual_concepts": {
                    self.concept_names[i]: float(orig_c[0, i].item()) for i in range(4)
                },
                "concept_deltas": {
                    self.concept_names[i]: 0.0 for i in range(4)
                },
                "original_probabilities": {
                    DIAGNOSIS_CLASSES.get(i, f"Class_{i}"): float(orig_probs[0, i].item())
                    for i in range(orig_probs.shape[1])
                },
                "counterfactual_probabilities": {
                    DIAGNOSIS_CLASSES.get(i, f"Class_{i}"): float(orig_probs[0, i].item())
                    for i in range(orig_probs.shape[1])
                },
                "explanation_text": f"The model already predicts {target_name} ({orig_probs[0, orig_pred_idx].item()*100:.1f}% confidence). No counterfactual modification needed.",
            }

        # Case 2: Optimize concept shift delta to flip prediction
        target_tensor = torch.tensor([target_class_idx], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        def _optimize(learning_rate, reg_weight, n_steps):
            delta = nn.Parameter(torch.zeros_like(orig_c))
            optimizer = optim.Adam([delta], lr=learning_rate)
            best_d = None
            best_l = float("inf")
            best_prob_d = None
            max_target_prob = -1.0
            found = False

            for _ in range(n_steps):
                optimizer.zero_grad()
                cf_c = torch.clamp(orig_c + delta, 0.0, 1.0)
                logits, probs = self.model.diagnosis_head(cf_c)

                t_prob = probs[0, target_class_idx].item()
                if t_prob > max_target_prob:
                    max_target_prob = t_prob
                    best_prob_d = delta.detach().clone()

                ce_loss = criterion(logits, target_tensor)
                sparsity_loss = torch.norm(delta, p=1)
                total_loss = ce_loss + reg_weight * sparsity_loss

                total_loss.backward()
                optimizer.step()

                pred = torch.argmax(probs, dim=-1).item()
                if pred == target_class_idx and total_loss.item() < best_l:
                    best_l = total_loss.item()
                    best_d = delta.detach().clone()
                    found = True

            return found, best_d if found else best_prob_d, max_target_prob

        # Primary optimization pass
        found_target, best_delta, max_prob = _optimize(lr, l1_reg, steps)

        # Adaptive fallback pass with higher learning rate and lower L1 restriction if needed
        if not found_target:
            found_fallback, fallback_delta, fallback_prob = _optimize(lr * 1.8, l1_reg * 0.2, steps + 50)
            if found_fallback or fallback_prob > max_prob:
                found_target = found_fallback
                best_delta = fallback_delta
                max_prob = fallback_prob

        if best_delta is None:
            best_delta = torch.zeros_like(orig_c)

        final_c = torch.clamp(orig_c + best_delta, 0.0, 1.0)
        with torch.no_grad():
            _, final_probs = self.model.diagnosis_head(final_c)

        final_pred_idx = torch.argmax(final_probs, dim=-1).item()
        achieved_name = DIAGNOSIS_CLASSES.get(final_pred_idx, "Unknown")

        changes = []
        for i, name in enumerate(self.concept_names):
            val_orig = orig_c[0, i].item()
            val_cf = final_c[0, i].item()
            diff = val_cf - val_orig
            if abs(diff) > 0.02:
                action = "reducing" if diff < 0 else "increasing"
                changes.append(f"{action} {name} from {val_orig:.2f} to {val_cf:.2f}")

        if found_target:
            change_desc = ", and ".join(changes) if changes else "minimal concept shift"
            explanation_text = (
                f"The diagnosis flips from {orig_name} to {target_name} by {change_desc} "
                f"(achieved {target_name} confidence: {final_probs[0, target_class_idx].item()*100:.1f}%)."
            )
        else:
            change_desc = ", and ".join(changes) if changes else "exploring concept boundaries"
            explanation_text = (
                f"A complete flip from {orig_name} to {target_name} requires crossing strong decision boundaries; "
                f"the closest counterfactual shift reaches {max_prob*100:.1f}% {target_name} probability by {change_desc}."
            )

        return {
            "target_class_idx": target_class_idx,
            "target_class_name": target_name,
            "success": found_target,
            "original_concepts": {
                self.concept_names[i]: float(orig_c[0, i].item()) for i in range(4)
            },
            "counterfactual_concepts": {
                self.concept_names[i]: float(final_c[0, i].item()) for i in range(4)
            },
            "concept_deltas": {
                self.concept_names[i]: float(best_delta[0, i].item()) for i in range(4)
            },
            "original_probabilities": {
                DIAGNOSIS_CLASSES.get(i, f"Class_{i}"): float(orig_probs[0, i].item())
                for i in range(orig_probs.shape[1])
            },
            "counterfactual_probabilities": {
                DIAGNOSIS_CLASSES.get(i, f"Class_{i}"): float(final_probs[0, i].item())
                for i in range(final_probs.shape[1])
            },
            "explanation_text": explanation_text,
        }

    def render_counterfactual_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        counterfactual_concepts: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render segmentation-constrained visual representation of counterfactual lesion.

        Applies visual modifications based on counterfactual concept values:
        - Border Irregularity: Smooth/regularize boundary if reduced
        - Color Variation: Homogenize internal color if reduced
        - Asymmetry: Mirror/smooth half of lesion if reduced
        - Diameter: Scale lesion region if modified

        Returns:
            Tuple of (counterfactual_image, counterfactual_mask)
        """
        import cv2

        cf_image = image.copy().astype(np.float32)
        mask_binary = ((mask > 0.5) | (mask > 127)).astype(np.uint8)

        irreg_score = counterfactual_concepts.get("Border Irregularity Score", 0.5)
        color_score = counterfactual_concepts.get("Color Variation Score", 0.5)
        asym_score = counterfactual_concepts.get("Asymmetry Score", 0.5)
        diam_score = counterfactual_concepts.get("Diameter Score", 0.5)

        # 1. Mask modification: smooth boundary if irregularity is low
        if irreg_score < 0.4:
            kernel_size = int(max(3, (0.5 - irreg_score) * 30))
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(mask_binary.astype(np.float32), (kernel_size, kernel_size), 0)
            cf_mask = (blurred > 0.5).astype(np.uint8)
        else:
            cf_mask = mask_binary.copy()

        # 2. Image modification: reduce color variation inside lesion
        if color_score < 0.4:
            lesion_idx = (cf_mask == 1)
            if np.any(lesion_idx):
                mean_color = np.mean(cf_image[lesion_idx], axis=0)
                alpha = float(1.0 - color_score)
                cf_image[lesion_idx] = (1.0 - alpha) * cf_image[lesion_idx] + alpha * mean_color

        # 3. Asymmetry modification: make lesion more symmetric by mirroring
        if asym_score < 0.3:
            rows = np.any(cf_mask, axis=1)
            cols = np.any(cf_mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # Extract lesion bounding box
                crop_mask = cf_mask[rmin:rmax+1, cmin:cmax+1]
                crop_img = cf_image[rmin:rmax+1, cmin:cmax+1].copy()
                # Mirror left half onto right half
                mid_col = crop_mask.shape[1] // 2
                if mid_col > 0:
                    w = crop_mask.shape[1]
                    right_len = w - mid_col
                    left_slice_mask = crop_mask[:, :right_len]
                    left_slice_img = crop_img[:, :right_len]
                    crop_mask[:, mid_col:] = np.fliplr(left_slice_mask)
                    crop_img[:, mid_col:] = np.fliplr(left_slice_img)
                    cf_mask[rmin:rmax+1, cmin:cmax+1] = crop_mask
                    cf_image[rmin:rmax+1, cmin:cmax+1] = crop_img

        # 4. Diameter modification: scale lesion region
        if diam_score < 0.3:
            # Shrink lesion by eroding the mask
            erode_iter = int(max(1, (0.4 - diam_score) * 10))
            kernel = np.ones((3, 3), np.uint8)
            cf_mask = cv2.erode(cf_mask, kernel, iterations=erode_iter)
            # Fill eroded border with background mean
            eroded_border = (mask_binary == 1) & (cf_mask == 0)
            if np.any(eroded_border):
                bg_pixels = cf_image[mask_binary == 0]
                if len(bg_pixels) > 0:
                    bg_mean = np.mean(bg_pixels, axis=0)
                    cf_image[eroded_border] = bg_mean

        cf_image_uint8 = np.clip(cf_image, 0, 255).astype(np.uint8)
        return cf_image_uint8, cf_mask

    def validate_counterfactual(
        self,
        original_concepts: Dict[str, float],
        counterfactual_concepts: Dict[str, float],
        target_class_name: str,
        achieved_class_name: str,
        max_l1_distance: float = 2.0,
    ) -> Dict[str, any]:
        """Validate counterfactual explanation for plausibility and correctness.

        Checks:
        (a) L1 distance between original and counterfactual concepts is within bounds
        (b) All counterfactual concept values remain in [0, 1]
        (c) Target class was actually achieved

        Args:
            original_concepts: Original concept scores dict.
            counterfactual_concepts: Counterfactual concept scores dict.
            target_class_name: Intended target diagnosis class.
            achieved_class_name: Actually predicted class after counterfactual.
            max_l1_distance: Maximum allowed L1 distance in concept space.

        Returns:
            Dictionary with validation results and any issues found.
        """
        issues = []

        # (a) L1 distance check
        deltas = []
        for key in original_concepts:
            orig_val = original_concepts.get(key, 0.5)
            cf_val = counterfactual_concepts.get(key, 0.5)
            deltas.append(abs(cf_val - orig_val))

        l1_distance = sum(deltas)
        if l1_distance > max_l1_distance:
            issues.append(f"L1 distance {l1_distance:.3f} exceeds maximum {max_l1_distance}")

        # (b) Concept range check
        for key, val in counterfactual_concepts.items():
            if val < 0.0 or val > 1.0:
                issues.append(f"Concept '{key}' out of range: {val:.4f}")

        # (c) Target class achieved
        target_achieved = (target_class_name == achieved_class_name)
        if not target_achieved:
            issues.append(f"Target class '{target_class_name}' not achieved; got '{achieved_class_name}'")

        # Sparsity: count how many concepts changed significantly
        significant_changes = sum(1 for d in deltas if d > 0.02)

        return {
            "is_valid": len(issues) == 0,
            "l1_distance": float(l1_distance),
            "target_achieved": target_achieved,
            "concepts_in_range": all(0.0 <= v <= 1.0 for v in counterfactual_concepts.values()),
            "sparsity": significant_changes,
            "total_concepts": len(deltas),
            "issues": issues,
        }

