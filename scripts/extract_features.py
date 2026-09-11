import os
import pandas as pd
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_extractor import extract_geometric_features

def main():
    print("Extracting features from all images...")
    index_csv = PROJECT_ROOT / "data" / "processed" / "isic2018_task3_index.csv"
    seg_csv = PROJECT_ROOT / "data" / "processed" / "isic2018_task1_segmentation_index.csv"
    
    if not index_csv.exists() or not seg_csv.exists():
        print("Indices not found. Run prepare_dataset.py first.")
        return
        
    diag_df = pd.read_csv(index_csv)
    seg_df = pd.read_csv(seg_csv)
    
    # Merge to get mask paths
    df = pd.merge(diag_df, seg_df[["image_id", "mask_path"]], on="image_id")
    
    records = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        features = extract_geometric_features(img, mask)
        features["image_id"] = row["image_id"]
        features["label"] = row["label"]
        features["label_index"] = row["label_index"]
        
        records.append(features)
        
    features_df = pd.DataFrame(records)
    out_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Extracted features saved to {out_path}")

if __name__ == "__main__":
    main()
