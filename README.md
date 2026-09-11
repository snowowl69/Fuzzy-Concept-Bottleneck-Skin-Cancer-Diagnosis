# Fuzzy Concept-Bottleneck Skin Cancer Diagnosis

## Project Status

The project is currently in the active development stage. The Streamlit prototype has been removed in favor of a robust Flask-based web application and an end-to-end AWS deployment plan.

## Objective

Build a research prototype that takes a dermoscopic skin-lesion image and produces:

1. A lesion segmentation mask (using U-Net).
2. Automatically extracted geometric and color features (ABCDE criteria).
3. Continuous fuzzy clinical concepts.
4. A diagnosis prediction across 7 ISIC classes.
5. A concept-level counterfactual explanation.

## Proposed Pipeline

Skin image
→ Preprocessing
→ Lesion segmentation (U-Net)
→ Geometric feature extraction
→ Adaptive fuzzy concept generator
→ Fuzzy concept bottleneck
→ Diagnosis Head
→ Counterfactual explanation generator

## Disclaimer

This project is a research prototype.
It is not a medical diagnostic tool and must not be used to make clinical decisions.
