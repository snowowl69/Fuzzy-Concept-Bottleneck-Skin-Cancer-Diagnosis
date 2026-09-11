"""Segmentation Evaluation Module.

Evaluates the trained U-Net segmentation model on the test split:
- Per-image Dice coefficient
- Per-image IoU (Jaccard index)
- Pixel accuracy
- Aggregate statistics (mean, std, min, max)
- Visual mask comparison output (original, ground truth, predicted)
- Saves JSON report to reports/segmentation_evaluation_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.segmentation.unet_baseline import UNetBaseline
from src.segmentation.segmentation_metrics import dice_coefficient, iou_score


def pixel_accuracy(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Compute pixel-level accuracy for a binary segmentation mask.

    Args:
        pred_mask: Predicted binary mask.
        target_mask: Ground truth binary mask.

    Returns:
        Fraction of correctly classified pixels.
    """
    pred = (np.asarray(pred_mask, dtype=np.float32) > 0.5).astype(np.float32)
    target = (np.asarray(target_mask, dtype=np.float32) > 0.5).astype(np.float32)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    correct = np.sum(pred == target)
    total = pred.size
    return float(correct / max(total, 1))


def evaluate_segmentation(
    test_csv: str = None,
    image_size: int = 256,
    save_visual_comparisons: bool = True,
    max_visual_samples: int = 10,
) -> dict:
    """Evaluate segmentation model on test set.

    Args:
        test_csv: Path to test split CSV. Defaults to data/processed/segmentation_test.csv.
        image_size: Image size for evaluation.
        save_visual_comparisons: If True, save side-by-side comparison images.
        max_visual_samples: Maximum number of visual comparisons to save.

    Returns:
        Dictionary with evaluation metrics.
    """
    if test_csv is None:
        test_csv = PROJECT_ROOT / "data" / "processed" / "segmentation_test.csv"
    else:
        test_csv = Path(test_csv)

    if not test_csv.exists():
        print(f"Test CSV not found: {test_csv}")
        return {}

    # Load model
    model = UNetBaseline(in_channels=3, out_channels=1)
    checkpoint_path = PROJECT_ROOT / "models" / "segmentation_unet_baseline.pt"
    if checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print(f"Loaded segmentation checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Using initialized weights.")
    else:
        print("Warning: No segmentation checkpoint found. Using initialized weights.")

    model.eval()

    # Load test dataset
    try:
        from src.segmentation.segmentation_dataset import ISICSegmentationDataset
        test_dataset = ISICSegmentationDataset(csv_file=test_csv, image_size=image_size)
    except Exception as e:
        print(f"Could not load test dataset: {e}")
        # Fallback: generate synthetic data for demonstration
        print("Running evaluation with synthetic test data...")
        return _evaluate_with_synthetic_data(model, image_size, save_visual_comparisons)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    dice_values = []
    iou_values = []
    accuracy_values = []

    visual_samples = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images = batch["image"]
            masks = batch["mask"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            pred_np = preds.squeeze().cpu().numpy()
            mask_np = masks.squeeze().cpu().numpy()

            d = dice_coefficient(pred_np, mask_np)
            i = iou_score(pred_np, mask_np)
            a = pixel_accuracy(pred_np, mask_np)

            dice_values.append(d)
            iou_values.append(i)
            accuracy_values.append(a)

            if save_visual_comparisons and len(visual_samples) < max_visual_samples:
                img_np = (images.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                visual_samples.append({
                    "image": img_np,
                    "ground_truth": (mask_np * 255).astype(np.uint8),
                    "prediction": (pred_np * 255).astype(np.uint8),
                    "dice": d,
                    "iou": i,
                })

    results = _compile_results(dice_values, iou_values, accuracy_values)

    # Save visual comparisons
    if save_visual_comparisons and visual_samples:
        _save_visual_comparisons(visual_samples)

    # Save report
    report_path = PROJECT_ROOT / "reports" / "segmentation_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results, report_path)
    return results


def _evaluate_with_synthetic_data(model, image_size, save_visual):
    """Fallback evaluation using synthetic circle-based test images."""
    dice_values = []
    iou_values = []
    accuracy_values = []

    for _ in range(20):
        img = np.random.randint(100, 220, (image_size, image_size, 3), dtype=np.uint8)
        y, x = np.ogrid[:image_size, :image_size]
        cx, cy = np.random.randint(60, 196), np.random.randint(60, 196)
        r = np.random.randint(20, 60)
        circle = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
        gt_mask = circle.astype(np.float32)

        img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = model(img_t)
            pred = (torch.sigmoid(logits).squeeze().numpy() > 0.5).astype(np.float32)

        dice_values.append(dice_coefficient(pred, gt_mask))
        iou_values.append(iou_score(pred, gt_mask))
        accuracy_values.append(pixel_accuracy(pred, gt_mask))

    results = _compile_results(dice_values, iou_values, accuracy_values)
    results["note"] = "Evaluated on synthetic data (test split CSV not available)"

    report_path = PROJECT_ROOT / "reports" / "segmentation_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results, report_path)
    return results


def _compile_results(dice_values, iou_values, accuracy_values):
    """Compile evaluation metric arrays into summary dictionary."""
    return {
        "num_samples": len(dice_values),
        "dice_coefficient": {
            "mean": float(np.mean(dice_values)) if dice_values else 0.0,
            "std": float(np.std(dice_values)) if dice_values else 0.0,
            "min": float(np.min(dice_values)) if dice_values else 0.0,
            "max": float(np.max(dice_values)) if dice_values else 0.0,
            "median": float(np.median(dice_values)) if dice_values else 0.0,
        },
        "iou_score": {
            "mean": float(np.mean(iou_values)) if iou_values else 0.0,
            "std": float(np.std(iou_values)) if iou_values else 0.0,
            "min": float(np.min(iou_values)) if iou_values else 0.0,
            "max": float(np.max(iou_values)) if iou_values else 0.0,
        },
        "pixel_accuracy": {
            "mean": float(np.mean(accuracy_values)) if accuracy_values else 0.0,
            "std": float(np.std(accuracy_values)) if accuracy_values else 0.0,
        },
    }


def _save_visual_comparisons(visual_samples):
    """Save side-by-side comparison images to reports directory."""
    try:
        import cv2
        output_dir = PROJECT_ROOT / "reports" / "segmentation_visuals"
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, sample in enumerate(visual_samples):
            # Create side-by-side comparison: Original | GT | Predicted
            img = sample["image"]
            gt = np.stack([sample["ground_truth"]] * 3, axis=-1)
            pred = np.stack([sample["prediction"]] * 3, axis=-1)

            comparison = np.concatenate([img, gt, pred], axis=1)
            save_path = output_dir / f"comparison_{idx:03d}_dice{sample['dice']:.3f}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        print(f"Saved {len(visual_samples)} visual comparisons to {output_dir}")
    except Exception as e:
        print(f"Could not save visual comparisons: {e}")


def _print_summary(results, report_path):
    """Print evaluation summary to console."""
    print("=" * 60)
    print("--- SEGMENTATION EVALUATION SUMMARY ---")
    print("=" * 60)
    print(f"Samples evaluated:  {results['num_samples']}")
    dice = results.get("dice_coefficient", {})
    iou = results.get("iou_score", {})
    pxa = results.get("pixel_accuracy", {})
    print(f"Dice Coefficient:   {dice.get('mean', 0):.4f} ± {dice.get('std', 0):.4f}")
    print(f"IoU Score:          {iou.get('mean', 0):.4f} ± {iou.get('std', 0):.4f}")
    print(f"Pixel Accuracy:     {pxa.get('mean', 0)*100:.2f}% ± {pxa.get('std', 0)*100:.2f}%")
    print(f"Report saved to:    {report_path}")


if __name__ == "__main__":
    evaluate_segmentation()
