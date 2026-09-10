from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "isic2018"

IMAGE_DIR = RAW_DIR / "task3_input"
GROUND_TRUTH_DIR = RAW_DIR / "task3_groundtruth"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "isic2018_task3_index.csv"

DIAGNOSIS_COLUMNS = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]

LABEL_TO_INDEX = {
    "MEL": 0,
    "NV": 1,
    "BCC": 2,
    "AKIEC": 3,
    "BKL": 4,
    "DF": 5,
    "VASC": 6,
}


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


def find_image_files() -> list[Path]:
    """Find all Task 3 JPEG images."""
    return sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg"}
    )


def one_hot_to_label(row: pd.Series) -> str:
    """Convert the seven one-hot diagnosis columns into one label."""
    active_classes = [
        column
        for column in DIAGNOSIS_COLUMNS
        if float(row[column]) == 1.0
    ]

    if len(active_classes) != 1:
        raise ValueError(
            f"Expected exactly one diagnosis for {row['image']}, "
            f"found {active_classes}"
        )

    return active_classes[0]


def main() -> None:
    print("=" * 60)
    print("CREATE ISIC 2018 TASK 3 DATASET INDEX")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Find images
    # ---------------------------------------------------------

    image_files = find_image_files()

    if len(image_files) != 10015:
        raise ValueError(
            f"Expected 10,015 images, found {len(image_files)}."
        )

    image_map = {
        path.stem: path
        for path in image_files
    }

    print(f"Images found: {len(image_map)}")

    # ---------------------------------------------------------
    # Load diagnosis CSV
    # ---------------------------------------------------------

    csv_path = find_diagnosis_csv()

    diagnosis_df = pd.read_csv(csv_path)

    if len(diagnosis_df) != 10015:
        raise ValueError(
            f"Expected 10,015 diagnosis records, "
            f"found {len(diagnosis_df)}."
        )

    print(f"Diagnosis records: {len(diagnosis_df)}")

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    required_columns = {
        "image",
        *DIAGNOSIS_COLUMNS,
    }

    missing_columns = required_columns - set(diagnosis_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing diagnosis columns: {sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # Create index rows
    # ---------------------------------------------------------

    rows = []

    for _, row in diagnosis_df.iterrows():
        image_id = str(row["image"])

        if image_id not in image_map:
            raise ValueError(
                f"Diagnosis record has no matching image: {image_id}"
            )

        label = one_hot_to_label(row)

        image_path = image_map[image_id]

        relative_image_path = image_path.relative_to(PROJECT_ROOT)

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(relative_image_path),
                "label": label,
                "label_index": LABEL_TO_INDEX[label],
            }
        )

    index_df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if len(index_df) != 10015:
        raise ValueError(
            f"Expected 10,015 index rows, found {len(index_df)}."
        )

    if index_df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs detected.")

    if index_df["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths detected.")

    if index_df["label"].isna().any():
        raise ValueError("Missing labels detected.")

    if not set(index_df["label"]).issubset(LABEL_TO_INDEX):
        raise ValueError("Unexpected diagnosis class detected.")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    index_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("DATASET INDEX")
    print("-" * 60)

    print(f"Rows: {len(index_df)}")
    print(f"Output: {OUTPUT_FILE}")

    print()
    print("Columns:")
    for column in index_df.columns:
        print(f"  - {column}")

    print()
    print("Class distribution:")

    counts = index_df["label"].value_counts()

    for label in DIAGNOSIS_COLUMNS:
        print(
            f"  {label:>5}: {int(counts.get(label, 0))}"
        )

    print()
    print("First 5 records:")
    print(index_df.head().to_string(index=False))

    print()
    print("=" * 60)
    print("DATASET INDEX CREATION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()