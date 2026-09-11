"""Feature Extraction Evaluation Module.

Evaluates the robustness and quality of geometric/color feature extraction:
- Feature extraction success rate (% of images processed without error)
- Mask validity check (% of masks with area > threshold)
- Feature range distribution analysis per feature
- Robustness test with different synthetic lesion shapes (circular, elongated, irregular)
- Saves report to reports/feature_evaluation_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_extractor import (
    extract_geometric_features,
    validate_features,
    normalize_features,
    compute_fourier_descriptors,
)


def _generate_synthetic_lesion(
    shape_type: str, image_size: int = 256
) -> tuple:
    """Generate a synthetic dermoscopy image and binary mask.

    Args:
        shape_type: One of 'circular', 'elongated', 'irregular', 'tiny', 'large'.
        image_size: Image dimensions (square).

    Returns:
        Tuple of (image_np, mask_np, shape_type)
    """
    img = np.random.randint(120, 220, (image_size, image_size, 3), dtype=np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    cx, cy = image_size // 2, image_size // 2
    y, x = np.ogrid[:image_size, :image_size]

    if shape_type == "circular":
        r = 50
        region = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
    elif shape_type == "elongated":
        # Elliptical shape with high aspect ratio
        rx, ry = 70, 25
        region = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1
    elif shape_type == "irregular":
        # Star-like irregular shape
        r_base = 40
        theta = np.arctan2(y - cy, x - cx)
        r_boundary = r_base + 15 * np.sin(5 * theta)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        region = dist <= r_boundary
    elif shape_type == "tiny":
        r = 8
        region = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
    elif shape_type == "large":
        r = 110
        region = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
    elif shape_type == "empty":
        region = np.zeros((image_size, image_size), dtype=bool)
    else:
        r = 45
        region = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2

    mask[region] = 1
    # Darken lesion area
    img[region] = (img[region] * 0.4 + np.array([80, 40, 30])).clip(0, 255).astype(np.uint8)

    return img, mask, shape_type


def evaluate_features(num_random_samples: int = 30) -> dict:
    """Run feature extraction evaluation.

    Args:
        num_random_samples: Number of random synthetic samples per shape type.

    Returns:
        Evaluation results dictionary.
    """
    shape_types = ["circular", "elongated", "irregular", "tiny", "large", "empty"]

    total_attempts = 0
    successful_extractions = 0
    failed_extractions = 0
    validation_passed = 0
    validation_failed = 0
    mask_valid_count = 0
    mask_invalid_count = 0

    feature_distributions = {}
    shape_results = {}
    fourier_results = {}

    for shape in shape_types:
        shape_stats = {
            "success_count": 0,
            "failure_count": 0,
            "validation_pass": 0,
            "validation_fail": 0,
            "feature_means": {},
            "feature_stds": {},
        }
        feature_values_per_shape = {}

        for _ in range(num_random_samples):
            total_attempts += 1
            img, mask, _ = _generate_synthetic_lesion(shape)

            try:
                features = extract_geometric_features(img, mask)
                successful_extractions += 1
                shape_stats["success_count"] += 1

                # Mask validity: area > 0
                area = features.get("area", 0.0)
                if area > 0:
                    mask_valid_count += 1
                else:
                    mask_invalid_count += 1

                # Feature validation
                is_valid, issues = validate_features(features)
                if is_valid:
                    validation_passed += 1
                    shape_stats["validation_pass"] += 1
                else:
                    validation_failed += 1
                    shape_stats["validation_fail"] += 1

                # Collect per-feature values
                for key, val in features.items():
                    if isinstance(val, (int, float)):
                        feature_values_per_shape.setdefault(key, []).append(float(val))
                        feature_distributions.setdefault(key, []).append(float(val))

                # Fourier descriptors
                try:
                    fd = compute_fourier_descriptors(mask)
                    fourier_results.setdefault(shape, []).append(fd.tolist())
                except Exception:
                    pass

            except Exception as e:
                failed_extractions += 1
                shape_stats["failure_count"] += 1

        # Compute per-shape statistics
        for key, vals in feature_values_per_shape.items():
            shape_stats["feature_means"][key] = float(np.mean(vals))
            shape_stats["feature_stds"][key] = float(np.std(vals))

        shape_results[shape] = shape_stats

    # Global feature range analysis
    feature_range_analysis = {}
    for key, vals in feature_distributions.items():
        feature_range_analysis[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }

    # Normalization test
    normalization_test = {"passed": True, "details": []}
    for shape in ["circular", "irregular"]:
        img, mask, _ = _generate_synthetic_lesion(shape)
        features = extract_geometric_features(img, mask)
        normed = normalize_features(features)
        for key, val in normed.items():
            if val < 0.0 or val > 1.0:
                normalization_test["passed"] = False
                normalization_test["details"].append(
                    f"{key} out of [0,1] after normalization: {val:.4f}"
                )

    results = {
        "total_attempts": total_attempts,
        "extraction_success_rate": float(successful_extractions / max(total_attempts, 1)),
        "extraction_failure_rate": float(failed_extractions / max(total_attempts, 1)),
        "validation_pass_rate": float(validation_passed / max(successful_extractions, 1)),
        "mask_validity_rate": float(mask_valid_count / max(mask_valid_count + mask_invalid_count, 1)),
        "feature_range_analysis": feature_range_analysis,
        "per_shape_results": shape_results,
        "normalization_test": normalization_test,
    }

    # Save report
    report_path = PROJECT_ROOT / "reports" / "feature_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("--- FEATURE EXTRACTION EVALUATION SUMMARY ---")
    print("=" * 60)
    print(f"Total attempts:           {total_attempts}")
    print(f"Extraction success rate:  {results['extraction_success_rate']*100:.1f}%")
    print(f"Validation pass rate:     {results['validation_pass_rate']*100:.1f}%")
    print(f"Mask validity rate:       {results['mask_validity_rate']*100:.1f}%")
    print(f"Normalization test:       {'PASSED' if normalization_test['passed'] else 'FAILED'}")
    print(f"\nPer-shape extraction success:")
    for shape, stats in shape_results.items():
        total = stats["success_count"] + stats["failure_count"]
        print(f"  {shape:12s}: {stats['success_count']}/{total} "
              f"(validation: {stats['validation_pass']}/{stats['success_count']})")
    print(f"\nReport saved to: {report_path}")

    return results


if __name__ == "__main__":
    evaluate_features()
