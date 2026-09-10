from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "isic2018"

IMAGE_DIR = RAW_DIR / "task3_input"
GROUND_TRUTH_DIR = RAW_DIR / "task3_groundtruth"

DIAGNOSIS_COLUMNS = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]


def find_diagnosis_csv() -> Path:
    """Find the Task 3 diagnosis CSV."""
    matches = list(
        GROUND_TRUTH_DIR.rglob(
            "ISIC2018_Task3_Training_GroundTruth.csv"
        )
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one diagnosis CSV, found {len(matches)}."
        )

    return matches[0]


def get_image_files() -> list[Path]:
    """Find all Task 3 JPEG images."""
    return sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg"}
    )


def convert_one_hot_to_label(row: pd.Series) -> str:
    """Convert one-hot diagnosis columns into a class name."""
    active_classes = [
        column
        for column in DIAGNOSIS_COLUMNS
        if float(row[column]) == 1.0
    ]

    if len(active_classes) != 1:
        raise ValueError(
            f"Expected exactly one active diagnosis class for "
            f"{row['image']}, found: {active_classes}"
        )

    return active_classes[0]


def main() -> None:
    print("=" * 60)
    print("TASK 3 IMAGE / LABEL VERIFICATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load images
    # ---------------------------------------------------------

    image_files = get_image_files()
    image_ids = {path.stem for path in image_files}

    print(f"Images found: {len(image_files)}")

    # ---------------------------------------------------------
    # Load diagnosis CSV
    # ---------------------------------------------------------

    csv_path = find_diagnosis_csv()
    df = pd.read_csv(csv_path)

    print(f"Diagnosis records: {len(df)}")
    print(f"Diagnosis CSV: {csv_path}")

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    expected_columns = {"image", *DIAGNOSIS_COLUMNS}

    missing_columns = expected_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # Normalize diagnosis IDs
    # ---------------------------------------------------------

    diagnosis_ids = set(df["image"].astype(str))

    # ---------------------------------------------------------
    # Compare IDs
    # ---------------------------------------------------------

    images_without_labels = image_ids - diagnosis_ids
    labels_without_images = diagnosis_ids - image_ids

    matched_ids = image_ids & diagnosis_ids

    print()
    print("-" * 60)
    print("ID MATCHING")
    print("-" * 60)

    print(f"Images with matching labels: {len(matched_ids)}")
    print(f"Images without labels: {len(images_without_labels)}")
    print(f"Labels without images: {len(labels_without_images)}")

    # ---------------------------------------------------------
    # Check duplicate IDs
    # ---------------------------------------------------------

    duplicate_ids = df[
        df["image"].astype(str).duplicated(keep=False)
    ]["image"].astype(str).unique()

    print()
    print("-" * 60)
    print("DUPLICATE CHECK")
    print("-" * 60)

    print(f"Duplicate diagnosis IDs: {len(duplicate_ids)}")

    # ---------------------------------------------------------
    # Convert labels
    # ---------------------------------------------------------

    matched_df = df[
        df["image"].astype(str).isin(matched_ids)
    ].copy()

    matched_df["label"] = matched_df.apply(
        convert_one_hot_to_label,
        axis=1,
    )

    # ---------------------------------------------------------
    # Class distribution
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("CLASS DISTRIBUTION")
    print("-" * 60)

    class_counts = matched_df["label"].value_counts()

    for class_name in DIAGNOSIS_COLUMNS:
        print(
            f"{class_name:>5}: "
            f"{int(class_counts.get(class_name, 0))}"
        )

    # ---------------------------------------------------------
    # Final checks
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("FINAL CHECKS")
    print("-" * 60)

    checks = {
        "10,015 images": len(image_files) == 10015,
        "10,015 diagnosis records": len(df) == 10015,
        "All images have labels": len(images_without_labels) == 0,
        "All labels have images": len(labels_without_images) == 0,
        "No duplicate diagnosis IDs": len(duplicate_ids) == 0,
        "All matched rows have one class": len(matched_df) == len(matched_ids),
    }

    all_passed = True

    for description, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {description}")

        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("=" * 60)
        print("TASK 3 DATASET VERIFICATION: PASS")
        print("=" * 60)
    else:
        print("=" * 60)
        print("TASK 3 DATASET VERIFICATION: FAILED")
        print("=" * 60)

        raise SystemExit(1)


if __name__ == "__main__":
    main()