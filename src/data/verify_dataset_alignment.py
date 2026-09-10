from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "isic2018"

TASK1_INPUT_DIR = RAW_DIR / "task1_input"
TASK1_GT_DIR = RAW_DIR / "task1_groundtruth"
TASK3_GT_DIR = RAW_DIR / "task3_groundtruth"


def find_files_recursively(directory: Path, extensions: set[str]) -> list[Path]:
    """Find files recursively with the requested extensions."""
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]


def get_image_ids() -> set[str]:
    """Return ISIC IDs from Task 1 JPEG images."""
    image_files = find_files_recursively(
        TASK1_INPUT_DIR,
        {".jpg", ".jpeg"},
    )

    image_ids = {path.stem for path in image_files}

    print(f"Task 1 images found: {len(image_files)}")

    return image_ids


def get_mask_ids() -> set[str]:
    """Return ISIC IDs from Task 1 segmentation masks."""
    mask_files = find_files_recursively(
        TASK1_GT_DIR,
        {".png"},
    )

    mask_ids = set()

    for path in mask_files:
        name = path.stem

        if name.endswith("_segmentation"):
            image_id = name.removesuffix("_segmentation")
            mask_ids.add(image_id)

    print(f"Task 1 segmentation masks found: {len(mask_ids)}")

    return mask_ids


def find_diagnosis_csv() -> Path:
    """Find the Task 3 diagnosis CSV."""
    csv_files = list(TASK3_GT_DIR.rglob("ISIC2018_Task3_Training_GroundTruth.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "Could not find ISIC2018_Task3_Training_GroundTruth.csv"
        )

    if len(csv_files) > 1:
        raise RuntimeError(
            f"Expected one diagnosis CSV, found {len(csv_files)}."
        )

    return csv_files[0]


def get_diagnosis_data() -> pd.DataFrame:
    """Load the Task 3 diagnosis CSV."""
    csv_path = find_diagnosis_csv()

    df = pd.read_csv(csv_path)

    print(f"Diagnosis CSV: {csv_path}")
    print(f"Diagnosis records found: {len(df)}")

    required_columns = {
        "image",
        "MEL",
        "NV",
        "BCC",
        "AKIEC",
        "BKL",
        "DF",
        "VASC",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing expected diagnosis columns: {sorted(missing_columns)}"
        )

    return df


def main() -> None:
    print("=" * 60)
    print("ISIC 2018 DATASET ALIGNMENT CHECK")
    print("=" * 60)

    image_ids = get_image_ids()
    mask_ids = get_mask_ids()
    diagnosis_df = get_diagnosis_data()

    diagnosis_ids = set(diagnosis_df["image"].astype(str))

    print()
    print("-" * 60)
    print("ID ALIGNMENT")
    print("-" * 60)

    images_without_masks = image_ids - mask_ids
    masks_without_images = mask_ids - image_ids

    images_without_diagnosis = image_ids - diagnosis_ids
    diagnosis_without_images = diagnosis_ids - image_ids

    matched_all = image_ids & mask_ids & diagnosis_ids

    print(f"Images with masks: {len(image_ids & mask_ids)}")
    print(f"Images with diagnosis labels: {len(image_ids & diagnosis_ids)}")
    print(f"Images with BOTH mask and diagnosis: {len(matched_all)}")

    print()
    print(f"Images without masks: {len(images_without_masks)}")
    print(f"Masks without images: {len(masks_without_images)}")
    print(f"Images without diagnosis labels: {len(images_without_diagnosis)}")
    print(f"Diagnosis records without Task 1 images: {len(diagnosis_without_images)}")

    print()
    print("-" * 60)
    print("DIAGNOSIS CLASS DISTRIBUTION")
    print("-" * 60)

    diagnosis_columns = [
        "MEL",
        "NV",
        "BCC",
        "AKIEC",
        "BKL",
        "DF",
        "VASC",
    ]

    matched_df = diagnosis_df[
        diagnosis_df["image"].astype(str).isin(matched_all)
    ].copy()

    print(f"Matched diagnosis records: {len(matched_df)}")
    print()

    for column in diagnosis_columns:
        count = int(matched_df[column].sum())
        print(f"{column:>5}: {count}")

    print()
    print("-" * 60)
    print("SAMPLE MATCHED IMAGE IDs")
    print("-" * 60)

    for image_id in sorted(matched_all)[:10]:
        print(image_id)

    print()
    print("=" * 60)
    print("ALIGNMENT CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()