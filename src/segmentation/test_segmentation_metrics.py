import numpy as np

from src.segmentation.segmentation_metrics import dice_coefficient, iou_score


def main() -> None:
    target = np.zeros((8, 8), dtype=np.float32)
    target[2:6, 2:6] = 1.0

    pred_perfect = target.copy()
    pred_ok = np.zeros_like(target)
    pred_ok[1:7, 1:7] = 1.0
    pred_wrong = np.ones_like(target)

    dice_perfect = dice_coefficient(pred_perfect, target)
    dice_ok = dice_coefficient(pred_ok, target)
    dice_wrong = dice_coefficient(pred_wrong, target)

    iou_perfect = iou_score(pred_perfect, target)
    iou_ok = iou_score(pred_ok, target)
    iou_wrong = iou_score(pred_wrong, target)

    assert 0.0 <= dice_perfect <= 1.0
    assert 0.0 <= iou_perfect <= 1.0
    assert dice_perfect > 0.9
    assert iou_perfect > 0.9
    assert dice_ok > dice_wrong
    assert iou_ok > iou_wrong

    empty_pred = np.zeros_like(target)
    empty_target = np.zeros_like(target)
    assert dice_coefficient(empty_pred, empty_target) == 1.0
    assert iou_score(empty_pred, empty_target) == 1.0

    print("SEGMENTATION METRICS TEST: PASS")


if __name__ == "__main__":
    main()
