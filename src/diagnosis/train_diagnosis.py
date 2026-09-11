"""Training script for Fuzzy Concept Bottleneck Model on Real Data.

Trains the Diagnosis Head and tunes Adaptive Fuzzy Concept Generator parameters
using extracted geometric features, with:
- Class imbalance weighting (inverse frequency / effective sample weighting)
- Multi-class ROC-AUC (One-vs-Rest macro & weighted)
- Per-class classification breakdown (Precision, Recall, F1, Support)
- Comprehensive evaluation reporting
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.diagnosis.concept_bottleneck import FuzzyConceptBottleneckModel
from src.diagnosis.diagnosis_head import DIAGNOSIS_CLASSES


def compute_class_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """Compute balanced class weights inversely proportional to class frequencies."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Avoid zero division with smoothing
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_concept_bottleneck_model(
    epochs: int = 50,
    lr: float = 0.005,
    batch_size: int = 32,
    use_class_weights: bool = True,
    concept_supervision_weight: float = 0.1,
) -> FuzzyConceptBottleneckModel:
    features_csv = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if not features_csv.exists():
        print(f"Features file {features_csv} not found! Run scripts/extract_features.py first.")
        return None

    df = pd.read_csv(features_csv)

    # Extract 4 features for fuzzy generator
    # ["asymmetry_index", "border_irregularity", "color_variation", "normalized_diameter"]
    feature_cols = ["asymmetry_index", "border_irregularity", "color_variation", "normalized_diameter"]
    X = df[feature_cols].values
    y = df["label_index"].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize 7-class model
    model = FuzzyConceptBottleneckModel(num_classes=7)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if use_class_weights:
        weights = compute_class_weights(y, num_classes=7)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Applied class weights: {np.round(weights.numpy(), 3)}")
    else:
        criterion = nn.CrossEntropyLoss()

    print(f"--- Training Fuzzy Concept Bottleneck Model ({epochs} epochs, 7 classes) ---")
    print(f"Concept supervision weight (lambda): {concept_supervision_weight}")
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_cls_loss = 0.0
        total_concept_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            concepts, logits, probs = model(batch_x)
            cls_loss = criterion(logits, batch_y)

            # Concept supervision: enforce consistency between predicted concepts
            # and clamped raw features as proxy ground-truth targets
            concept_sup_loss = model.concept_generator.compute_concept_supervision_loss(
                concepts, batch_x
            )
            loss = cls_loss + concept_supervision_weight * concept_sup_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_cls_loss += cls_loss.item() * batch_x.size(0)
            total_concept_loss += concept_sup_loss.item() * batch_x.size(0)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        if epoch % 10 == 0 or epoch == epochs:
            acc = accuracy_score(all_labels, all_preds)
            avg_cls = total_cls_loss / len(dataset)
            avg_cpt = total_concept_loss / len(dataset)
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {total_loss/len(dataset):.4f} "
                  f"(cls: {avg_cls:.4f}, concept: {avg_cpt:.4f}) | Accuracy: {acc * 100:.2f}%")

    # Comprehensive Evaluation
    model.eval()
    with torch.no_grad():
        _, _, final_probs_t = model(X_tensor)
        final_probs = final_probs_t.cpu().numpy()
        final_preds = np.argmax(final_probs, axis=-1)

    acc = accuracy_score(y, final_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y, final_preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y, final_preds, average="weighted", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y, final_preds, average=None, labels=list(range(7)), zero_division=0
    )

    # Multi-class ROC-AUC (One-vs-Rest)
    try:
        roc_auc_macro = roc_auc_score(y, final_probs, multi_class="ovr", average="macro")
        roc_auc_weighted = roc_auc_score(y, final_probs, multi_class="ovr", average="weighted")
    except Exception as e:
        print(f"ROC-AUC Calculation Notice: {e}")
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0

    cm = confusion_matrix(y, final_preds, labels=list(range(7)))

    print("\n" + "=" * 60)
    print("--- COMPREHENSIVE DIAGNOSIS EVALUATION METRICS ---")
    print("=" * 60)
    print(f"Accuracy:            {acc * 100:.2f}%")
    print(f"Macro Precision:     {macro_p * 100:.2f}%")
    print(f"Macro Recall:        {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:      {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:   {weighted_f1 * 100:.2f}%")
    print(f"ROC-AUC (Macro OvR): {roc_auc_macro:.4f}")
    print(f"ROC-AUC (Weighted):  {roc_auc_weighted:.4f}")

    print("\n--- Per-Class Performance Breakdown ---")
    class_report_dict = {}
    for i in range(7):
        name = DIAGNOSIS_CLASSES.get(i, f"Class_{i}")
        print(f"  [{i}] {name:32s} | P: {per_class_p[i]*100:5.2f}% | R: {per_class_r[i]*100:5.2f}% | F1: {per_class_f1[i]*100:5.2f}% | Support: {int(per_class_support[i])}")
        class_report_dict[name] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }

    print("\nConfusion Matrix (Rows = True, Cols = Pred):")
    print(cm)

    # Save model checkpoint
    save_path = PROJECT_ROOT / "models" / "concept_bottleneck.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    print(f"\nModel checkpoint saved to {save_path}")

    # Save detailed evaluation report
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_report = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro_ovr": float(roc_auc_macro),
        "roc_auc_weighted_ovr": float(roc_auc_weighted),
        "per_class": class_report_dict,
        "confusion_matrix": cm.tolist(),
    }
    with open(report_dir / "diagnosis_evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"Evaluation report written to {report_dir / 'diagnosis_evaluation_report.json'}")

    return model


if __name__ == "__main__":
    train_concept_bottleneck_model()
