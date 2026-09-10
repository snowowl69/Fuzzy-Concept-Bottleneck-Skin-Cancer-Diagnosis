from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "isic2018_task1_segmentation_index.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = OUTPUT_DIR / "segmentation_train.csv"
VAL_PATH = OUTPUT_DIR / "segmentation_validation.csv"
TEST_PATH = OUTPUT_DIR / "segmentation_test.csv"
RANDOM_SEED = 42


def main() -> None:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Segmentation index not found: {INDEX_PATH}")

    df = pd.read_csv(INDEX_PATH)
    if len(df) != 2594:
        raise ValueError(f"Expected 2,594 records, found {len(df)}.")

    required_columns = {"image_id", "image_path", "mask_path"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs detected in segmentation index.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    total = len(df)
    if len(train_df) + len(val_df) + len(test_df) != total:
        raise ValueError("Split sizes do not add up to original dataset size.")

    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    if train_ids & val_ids:
        raise ValueError("Data leakage: train/validation overlap detected.")
    if train_ids & test_ids:
        raise ValueError("Data leakage: train/test overlap detected.")
    if val_ids & test_ids:
        raise ValueError("Data leakage: validation/test overlap detected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Train count:", len(train_df))
    print("Validation count:", len(val_df))
    print("Test count:", len(test_df))
    print("Total count:", total)
    print("Train ∩ Validation:", len(train_ids & val_ids))
    print("Train ∩ Test:", len(train_ids & test_ids))
    print("Validation ∩ Test:", len(val_ids & test_ids))
    print(f"Train path: {TRAIN_PATH}")
    print(f"Validation path: {VAL_PATH}")
    print(f"Test path: {TEST_PATH}")


if __name__ == "__main__":
    main()
