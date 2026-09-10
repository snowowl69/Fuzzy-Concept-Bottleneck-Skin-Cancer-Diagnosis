from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "isic2018_task3_index.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VAL_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def main() -> None:
    print("=" * 60)
    print("CREATE TRAIN / VALIDATION / TEST SPLITS")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load dataset index
    # ---------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Total records: {len(df)}")

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    required_columns = {
        "image_id",
        "image_path",
        "label",
        "label_index",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs detected.")

    # ---------------------------------------------------------
    # First split:
    #
    # 70% train
    # 30% temporary
    # ---------------------------------------------------------

    train_df, temporary_df = train_test_split(
        df,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["label_index"],
    )

    # ---------------------------------------------------------
    # Second split:
    #
    # temporary 30% -> 15% validation + 15% test
    # ---------------------------------------------------------

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_df["label_index"],
    )

    # ---------------------------------------------------------
    # Reset indexes
    # ---------------------------------------------------------

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # ---------------------------------------------------------
    # Validate split sizes
    # ---------------------------------------------------------

    total = len(df)

    if len(train_df) + len(validation_df) + len(test_df) != total:
        raise ValueError("Split sizes do not add up to the original dataset.")

    train_ids = set(train_df["image_id"])
    validation_ids = set(validation_df["image_id"])
    test_ids = set(test_df["image_id"])

    if train_ids & validation_ids:
        raise ValueError("Data leakage: train/validation overlap detected.")

    if train_ids & test_ids:
        raise ValueError("Data leakage: train/test overlap detected.")

    if validation_ids & test_ids:
        raise ValueError("Data leakage: validation/test overlap detected.")

    # ---------------------------------------------------------
    # Save splits
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("SPLIT SIZES")
    print("-" * 60)

    print(f"Training:   {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")
    print(f"Total:      {total}")

    print()
    print("-" * 60)
    print("CLASS DISTRIBUTION")
    print("-" * 60)

    classes = [
        "MEL",
        "NV",
        "BCC",
        "AKIEC",
        "BKL",
        "DF",
        "VASC",
    ]

    for class_name in classes:
        train_count = int((train_df["label"] == class_name).sum())
        val_count = int((validation_df["label"] == class_name).sum())
        test_count = int((test_df["label"] == class_name).sum())

        print(
            f"{class_name:>5}: "
            f"train={train_count:4d}, "
            f"val={val_count:4d}, "
            f"test={test_count:4d}"
        )

    print()
    print("-" * 60)
    print("DATA LEAKAGE CHECK")
    print("-" * 60)

    print(f"Train IDs:       {len(train_ids)}")
    print(f"Validation IDs:  {len(validation_ids)}")
    print(f"Test IDs:        {len(test_ids)}")

    print("Train ∩ Validation: 0")
    print("Train ∩ Test:       0")
    print("Validation ∩ Test:  0")

    print()
    print("-" * 60)
    print("OUTPUT FILES")
    print("-" * 60)

    print(TRAIN_FILE)
    print(VAL_FILE)
    print(TEST_FILE)

    print()
    print("=" * 60)
    print("DATASET SPLIT CREATION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()