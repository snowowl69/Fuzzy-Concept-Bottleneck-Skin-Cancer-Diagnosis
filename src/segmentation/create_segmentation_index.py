from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "isic2018" / "task1_input"
MASK_ROOT = PROJECT_ROOT / "data" / "raw" / "isic2018" / "task1_groundtruth"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "isic2018_task1_segmentation_index.csv"


def find_images() -> list[Path]:
    image_dir = IMAGE_ROOT / "ISIC2018_Task1-2_Training_Input"
    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )


def find_masks() -> list[Path]:
    mask_dir = MASK_ROOT / "ISIC2018_Task1_Training_GroundTruth"
    return sorted(mask_dir.rglob("*_segmentation.png"))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_files = find_images()
    mask_files = find_masks()

    print(f"Images found: {len(image_files)}")
    print(f"Masks found: {len(mask_files)}")

    if len(image_files) != 2594:
        raise ValueError(f"Expected 2,594 Task 1 images, found {len(image_files)}.")

    image_map = {path.stem: path for path in image_files}
    mask_map = {path.stem.replace("_segmentation", ""): path for path in mask_files}

    unmatched_images = sorted(set(image_map) - set(mask_map))
    unmatched_masks = sorted(set(mask_map) - set(image_map))

    if unmatched_images or unmatched_masks:
        raise ValueError(
            f"Unmatched records detected: images={len(unmatched_images)}, masks={len(unmatched_masks)}."
        )

    duplicate_image_ids = [value for value in image_map if list(image_map).count(value) > 1]
    if duplicate_image_ids:
        raise ValueError(f"Duplicate image IDs detected: {duplicate_image_ids[:10]}")

    duplicate_mask_paths = [path for path in mask_files if mask_files.count(path) > 1]
    if duplicate_mask_paths:
        raise ValueError(f"Duplicate mask paths detected: {duplicate_mask_paths[:10]}")

    rows = []
    for image_id, image_path in sorted(image_map.items()):
        mask_path = mask_map[image_id]

        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask path does not exist: {mask_path}")

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
            }
        )

    index_df = pd.DataFrame(rows)

    if len(index_df) != 2594:
        raise ValueError(f"Expected 2,594 records, found {len(index_df)}.")
    if index_df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs found in index.")
    if index_df["mask_path"].duplicated().any():
        raise ValueError("Duplicate mask paths found in index.")

    index_df = index_df.sort_values("image_id", kind="mergesort").reset_index(drop=True)
    index_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("-" * 60)
    print("SEGMENTATION INDEX")
    print("-" * 60)
    print(f"Rows: {len(index_df)}")
    print(f"Output: {OUTPUT_PATH}")
    print("Columns:")
    for column in index_df.columns:
        print(f"  - {column}")
    print()
    print("IMAGE / MASK MATCHING: PASS")


if __name__ == "__main__":
    main()
