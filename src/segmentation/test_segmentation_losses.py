import torch

from src.segmentation.segmentation_losses import bce_dice_loss, dice_score


def main() -> None:
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:40, 20:40] = 1.0

    logits = torch.randn(2, 1, 64, 64)
    logits *= 0.1
    logits[:, :, 20:40, 20:40] += 2.0

    loss = bce_dice_loss(logits, target)
    assert torch.isfinite(loss), "Loss is not finite."
    assert loss.requires_grad, "Loss does not require gradients."

    dice = dice_score(logits, target)
    assert torch.all((dice >= 0.0) & (dice <= 1.0)), dice

    perfect_pred = torch.zeros_like(logits)
    perfect_pred[:, :, 20:40, 20:40] = 10.0
    perfect_dice = dice_score(perfect_pred, target)
    assert (perfect_dice.mean() > 0.8).item(), perfect_dice.mean().item()

    wrong_pred = torch.zeros_like(logits)
    wrong_pred[:, :, 0:10, 0:10] = 10.0
    wrong_dice = dice_score(wrong_pred, target)
    assert (wrong_dice.mean() < perfect_dice.mean()).item(), (wrong_dice.mean(), perfect_dice.mean())

    print("SEGMENTATION LOSS TEST: PASS")


if __name__ == "__main__":
    main()
