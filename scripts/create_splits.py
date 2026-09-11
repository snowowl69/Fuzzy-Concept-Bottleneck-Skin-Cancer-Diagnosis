import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    master_csv = OUTPUT_DIR / "isic2018_master_index.csv"
    if not master_csv.exists():
        print("Master CSV not found. Run prepare_dataset.py first.")
        return
        
    df = pd.read_csv(master_csv)
    
    # 70/15/15 split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label_index"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label_index"])
    
    # Segmentation splits
    train_df[["image_id", "image_path", "mask_path"]].to_csv(OUTPUT_DIR / "segmentation_train.csv", index=False)
    val_df[["image_id", "image_path", "mask_path"]].to_csv(OUTPUT_DIR / "segmentation_validation.csv", index=False)
    test_df[["image_id", "image_path", "mask_path"]].to_csv(OUTPUT_DIR / "segmentation_test.csv", index=False)
    
    # Diagnosis splits
    train_df[["image_id", "image_path", "label", "label_index"]].to_csv(OUTPUT_DIR / "diagnosis_train.csv", index=False)
    val_df[["image_id", "image_path", "label", "label_index"]].to_csv(OUTPUT_DIR / "diagnosis_validation.csv", index=False)
    test_df[["image_id", "image_path", "label", "label_index"]].to_csv(OUTPUT_DIR / "diagnosis_test.csv", index=False)

    print(f"Created splits: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")

if __name__ == "__main__":
    main()
