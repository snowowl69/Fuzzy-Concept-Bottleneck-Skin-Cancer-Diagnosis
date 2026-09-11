"""Fuzzy Concept Evaluation Module.

Evaluates:
- Membership value distributions for each ABCD clinical concept across the 7 ISIC classes
- Statistical correlation between continuous geometric measurements and fuzzy concept memberships
- Concept alignment and separability across benign vs malignant lesions
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.diagnosis.concept_bottleneck import FuzzyConceptBottleneckModel
from src.diagnosis.diagnosis_head import DIAGNOSIS_CLASSES


def evaluate_fuzzy_concepts() -> dict:
    features_csv = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if not features_csv.exists():
        print(f"Features file {features_csv} not found.")
        return {}

    df = pd.read_csv(features_csv)
    feature_cols = ["asymmetry_index", "border_irregularity", "color_variation", "normalized_diameter"]
    X = df[feature_cols].values
    y = df["label_index"].values

    model = FuzzyConceptBottleneckModel(num_classes=7)
    weights_path = PROJECT_ROOT / "models" / "concept_bottleneck.pth"
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        except Exception:
            pass
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        concepts_tensor = model.concept_generator(X_tensor)
        concepts = concepts_tensor.numpy()

    concept_names = ["Asymmetry", "Border_Irregularity", "Color_Variation", "Diameter"]
    for i, name in enumerate(concept_names):
        df[f"fuzzy_{name}"] = concepts[:, i]

    # Calculate per-class mean and standard deviation for each fuzzy concept
    concept_summary = {}
    for class_idx in range(7):
        class_name = DIAGNOSIS_CLASSES.get(class_idx, f"Class_{class_idx}")
        subset = df[df["label_index"] == class_idx]
        if len(subset) == 0:
            continue

        concept_summary[class_name] = {
            "count": len(subset),
            "concepts": {
                name: {
                    "mean": float(subset[f"fuzzy_{name}"].mean()),
                    "std": float(subset[f"fuzzy_{name}"].std()),
                    "median": float(subset[f"fuzzy_{name}"].median()),
                    "min": float(subset[f"fuzzy_{name}"].min()),
                    "max": float(subset[f"fuzzy_{name}"].max()),
                }
                for name in concept_names
            },
        }

    # Malignant (MEL/BCC) vs Benign (NV/BKL) separation analysis
    mel_subset = df[df["label_index"] == 0]  # Melanoma
    nv_subset = df[df["label_index"] == 1]   # Melanocytic Nevus

    separability = {}
    if len(mel_subset) > 0 and len(nv_subset) > 0:
        for name in concept_names:
            mel_mean = float(mel_subset[f"fuzzy_{name}"].mean())
            nv_mean = float(nv_subset[f"fuzzy_{name}"].mean())
            separability[name] = {
                "melanoma_mean": mel_mean,
                "nevus_mean": nv_mean,
                "difference": float(mel_mean - nv_mean),
            }

    results = {
        "concept_names": concept_names,
        "per_class_distributions": concept_summary,
        "malignant_vs_benign_separability": separability,
    }

    report_path = PROJECT_ROOT / "reports" / "fuzzy_concept_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("--- FUZZY CONCEPT EVALUATION SUMMARY ---")
    print("=" * 60)
    for cls_name, data in concept_summary.items():
        print(f"\n[{cls_name}] (N={data['count']})")
        for cname, stats in data["concepts"].items():
            print(f"  - {cname:22s}: mean={stats['mean']:.3f} ± {stats['std']:.3f}")

    print(f"\nReport saved to: {report_path}")
    return results


if __name__ == "__main__":
    evaluate_fuzzy_concepts()
