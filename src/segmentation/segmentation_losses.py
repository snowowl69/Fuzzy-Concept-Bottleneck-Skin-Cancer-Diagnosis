from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice coefficient for binary segmentation masks. Input tensors should be logits or probabilities."""
    if pred_logits.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred_logits={pred_logits.shape}, target={target.shape}")

    pred = torch.sigmoid(pred_logits)
    pred = (pred > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss computed from sigmoid probabilities. Safe for empty lesion masks."""
    if pred_logits.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred_logits={pred_logits.shape}, target={target.shape}")

    pred = torch.sigmoid(pred_logits)
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Combined binary cross-entropy and Dice loss."""
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="mean")
    dice = dice_loss(pred_logits, target, eps=eps)
    return bce + dice


def perfect_prediction_dice_score() -> float:
    return 1.0
