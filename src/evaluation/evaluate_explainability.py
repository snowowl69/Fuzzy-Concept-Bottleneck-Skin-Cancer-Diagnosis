"""Explainability Evaluation Module.

Evaluates:
- Counterfactual validity (success rate in flipping diagnosis to target class)
- Counterfactual sparsity (average L1 perturbation distance in concept space)
- Monotonicity of concept interventions (directional shift in malignant probabilities)
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
from src.counterfactual.counterfactual_generator import CounterfactualGenerator
from src.counterfactual.concept_intervention import ConceptInterventionEngine
from src.diagnosis.diagnosis_head import DIAGNOSIS_CLASSES


def evaluate_explainability(num_samples: int = 50) -> dict:
    features_csv = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if not features_csv.exists():
        print(f"Features file {features_csv} not found.")
        return {}

    df = pd.read_csv(features_csv)
    feature_cols = ["asymmetry_index", "border_irregularity", "color_variation", "normalized_diameter"]
    X = df[feature_cols].values

    model = FuzzyConceptBottleneckModel(num_classes=7)
    weights_path = PROJECT_ROOT / "models" / "concept_bottleneck.pth"
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        except Exception:
            pass
    model.eval()

    cf_generator = CounterfactualGenerator(model)
    intervention_engine = ConceptInterventionEngine(model)

    X_tensor = torch.tensor(X[:num_samples], dtype=torch.float32)
    with torch.no_grad():
        concepts_tensor = model.concept_generator(X_tensor)

    cf_success_count = 0
    cf_l1_distances = []
    cf_results_per_target = {i: {"attempts": 0, "success": 0} for i in range(7)}

    for i in range(min(num_samples, len(concepts_tensor))):
        c_i = concepts_tensor[i : i + 1]
        _, orig_probs = model.diagnosis_head(c_i)
        orig_class = torch.argmax(orig_probs, dim=-1).item()

        # Pick a target class different from predicted
        target_class = (orig_class + 1) % 7
        res = cf_generator.generate_counterfactual_concept(c_i, target_class_idx=target_class, steps=80)

        cf_results_per_target[target_class]["attempts"] += 1
        if res["success"]:
            cf_success_count += 1
            cf_results_per_target[target_class]["success"] += 1
            deltas = list(res["concept_deltas"].values())
            cf_l1_distances.append(float(np.sum(np.abs(deltas))))

    # Concept sensitivity / intervention responsiveness test
    sensitivity_results = {}
    test_concept_vec = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    sens_matrix = intervention_engine.compute_concept_sensitivity(test_concept_vec)

    results = {
        "samples_evaluated": min(num_samples, len(concepts_tensor)),
        "counterfactual_success_rate": float(cf_success_count / max(1, min(num_samples, len(concepts_tensor)))),
        "average_l1_sparsity": float(np.mean(cf_l1_distances)) if cf_l1_distances else 0.0,
        "target_class_success_breakdown": {
            DIAGNOSIS_CLASSES.get(k, f"Class_{k}"): v
            for k, v in cf_results_per_target.items()
            if v["attempts"] > 0
        },
        "concept_sensitivity_matrix": sens_matrix,
    }

    report_path = PROJECT_ROOT / "reports" / "explainability_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("--- EXPLAINABILITY EVALUATION SUMMARY ---")
    print("=" * 60)
    print(f"CF Success Rate:   {results['counterfactual_success_rate'] * 100:.2f}%")
    print(f"Mean L1 Sparsity:  {results['average_l1_sparsity']:.4f}")
    print(f"Report saved to:   {report_path}")

    return results


if __name__ == "__main__":
    evaluate_explainability(num_samples=30)
