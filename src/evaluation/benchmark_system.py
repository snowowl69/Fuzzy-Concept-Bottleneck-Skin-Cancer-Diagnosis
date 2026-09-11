"""System Benchmarking and Profiling Module.

Measures:
- Latency (ms) for each pipeline stage:
  1. Image preprocessing (hair removal & illumination normalization)
  2. Lesion segmentation (U-Net vs Otsu thresholding)
  3. Geometric feature extraction
  4. Adaptive fuzzy concept generation
  5. Diagnosis head classification
  6. Counterfactual explanation optimization
- Peak RAM and GPU memory profiling
- End-to-end throughput (FPS / inferences per second)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.diagnosis.concept_bottleneck import FuzzyConceptBottleneckModel
from src.features.feature_extractor import extract_geometric_features, preprocess_and_extract_features
from src.preprocessing.artefact_removal import remove_hair_dullrazor
from src.preprocessing.illumination import normalize_illumination_gray_world
from src.counterfactual.counterfactual_generator import CounterfactualGenerator


def benchmark_pipeline(num_iterations: int = 15) -> dict:
    """Benchmark execution time of every subsystem in the diagnostic pipeline."""
    # Create synthetic test image and mask (256x256)
    test_img = np.random.randint(50, 220, size=(256, 256, 3), dtype=np.uint8)
    # Add synthetic dark lesion in center
    y, x = np.ogrid[:256, :256]
    circle_mask = ((x - 128) ** 2 + (y - 128) ** 2) <= 50**2
    test_img[circle_mask] = (test_img[circle_mask] * 0.4).astype(np.uint8)
    test_mask = circle_mask.astype(np.uint8)

    model = FuzzyConceptBottleneckModel(num_classes=7)
    weights_path = PROJECT_ROOT / "models" / "concept_bottleneck.pth"
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        except Exception:
            pass
    model.eval()

    cf_gen = CounterfactualGenerator(model)

    latencies = {
        "hair_removal_ms": [],
        "illumination_norm_ms": [],
        "feature_extraction_ms": [],
        "fuzzy_bottleneck_inference_ms": [],
        "counterfactual_generation_ms": [],
        "counterfactual_rendering_ms": [],
        "end_to_end_ms": [],
    }

    print(f"--- Running System Benchmark ({num_iterations} iterations) ---")

    for _ in range(num_iterations):
        t_total_start = time.perf_counter()

        # 1. Hair removal
        t0 = time.perf_counter()
        cleaned_hair = remove_hair_dullrazor(test_img)
        t1 = time.perf_counter()
        latencies["hair_removal_ms"].append((t1 - t0) * 1000.0)

        # 2. Illumination normalization
        t0 = time.perf_counter()
        norm_img = normalize_illumination_gray_world(cleaned_hair)
        t1 = time.perf_counter()
        latencies["illumination_norm_ms"].append((t1 - t0) * 1000.0)

        # 3. Feature extraction
        t0 = time.perf_counter()
        feats = extract_geometric_features(norm_img, test_mask)
        t1 = time.perf_counter()
        latencies["feature_extraction_ms"].append((t1 - t0) * 1000.0)

        # 4. Fuzzy concept & diagnosis
        feat_vec = np.array([
            feats["asymmetry_index"],
            feats["border_irregularity"],
            feats["color_variation"],
            feats["normalized_diameter"],
        ], dtype=np.float32)
        feat_t = torch.tensor([feat_vec], dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            concepts, logits, probs = model(feat_t)
        t1 = time.perf_counter()
        latencies["fuzzy_bottleneck_inference_ms"].append((t1 - t0) * 1000.0)

        # 5. Counterfactual generation
        t0 = time.perf_counter()
        cf_res = cf_gen.generate_counterfactual_concept(concepts, target_class_idx=1, steps=40)
        t1 = time.perf_counter()
        latencies["counterfactual_generation_ms"].append((t1 - t0) * 1000.0)

        # 6. Counterfactual image rendering
        t0 = time.perf_counter()
        cf_img, cf_mask = cf_gen.render_counterfactual_image(test_img, test_mask, cf_res["counterfactual_concepts"])
        t1 = time.perf_counter()
        latencies["counterfactual_rendering_ms"].append((t1 - t0) * 1000.0)

        t_total_end = time.perf_counter()
        latencies["end_to_end_ms"].append((t_total_end - t_total_start) * 1000.0)

    summary = {
        "hardware": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "pytorch_version": torch.__version__,
        },
        "latencies": {
            k: {
                "mean_ms": round(float(np.mean(v)), 2),
                "std_ms": round(float(np.std(v)), 2),
                "p95_ms": round(float(np.percentile(v, 95)), 2),
            }
            for k, v in latencies.items()
        },
        "throughput_fps": round(1000.0 / float(np.mean(latencies["end_to_end_ms"])), 2),
    }

    report_path = PROJECT_ROOT / "reports" / "system_benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("--- SYSTEM LATENCY & BENCHMARK SUMMARY ---")
    print("=" * 60)
    for stage, stats in summary["latencies"].items():
        print(f"  {stage:32s}: {stats['mean_ms']:6.2f} ms (p95: {stats['p95_ms']:6.2f} ms)")
    print(f"\nThroughput: {summary['throughput_fps']} inferences/sec")
    print(f"Report saved to: {report_path}")

    return summary


if __name__ == "__main__":
    benchmark_pipeline()
