from __future__ import annotations

import numpy as np


def dice_coefficient(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-8) -> float:
    """Compute the Dice coefficient for a single binary segmentation mask.

    Empty masks are handled safely: if both masks are empty, return 1.0; if one is empty,
    the coefficient is 0.0 for the non-empty mask unless both are empty.
    """
    pred = np.asarray(pred_mask, dtype=np.float32)
    target = np.asarray(target_mask, dtype=np.float32)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)

    if pred_binary.sum() == 0 and target_binary.sum() == 0:
        return 1.0
    if pred_binary.sum() == 0 or target_binary.sum() == 0:
        return 0.0

    intersection = np.sum(pred_binary * target_binary)
    denom = np.sum(pred_binary) + np.sum(target_binary)
    return float((2.0 * intersection + eps) / (denom + eps))


def iou_score(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-8) -> float:
    """Compute IoU/Jaccard score for a single binary segmentation mask."""
    pred = np.asarray(pred_mask, dtype=np.float32)
    target = np.asarray(target_mask, dtype=np.float32)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)

    if pred_binary.sum() == 0 and target_binary.sum() == 0:
        return 1.0
    if pred_binary.sum() == 0 or target_binary.sum() == 0:
        return 0.0

    intersection = np.sum(pred_binary * target_binary)
    union = np.sum((pred_binary + target_binary) > 0)
    return float((intersection + eps) / (union + eps))
