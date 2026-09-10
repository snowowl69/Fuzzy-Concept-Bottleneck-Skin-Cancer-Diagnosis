from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "isic2018" / "task1_input" / "ISIC2018_Task1-2_Training_Input"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "isic2018" / "task1_groundtruth" / "ISIC2018_Task1_Training_GroundTruth"


def find_matching_mask(image_path: Path) -> Path:
    image_id = image_path.stem
    candidates = sorted(MASK_DIR.glob(f"{image_id}_segmentation.png"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one mask for {image_id}, found {len(candidates)}."
        )
    return candidates[0]


def main() -> None:
    jpg_files = sorted(IMAGE_DIR.glob("*.jpg"))
    if not jpg_files:
        raise FileNotFoundError(f"No JPEG files found under {IMAGE_DIR}.")

    sample_image = jpg_files[0]
    sample_mask = find_matching_mask(sample_image)

    image = Image.open(sample_image).convert("RGB")
    mask = Image.open(sample_mask)

    image_array = np.asarray(image)
    mask_array = np.asarray(mask)

    image_h, image_w = image_array.shape[:2]
    mask_h, mask_w = mask_array.shape[:2]

    print(f"Image ID: {sample_image.stem}")
    print(f"Image path: {sample_image}")
    print(f"Mask path: {sample_mask}")
    print(f"Image dimensions: {image.size}")
    print(f"Mask dimensions: {mask.size}")

    if (image_h, image_w) != (mask_h, mask_w):
        raise AssertionError(
            f"Dimension check failed: image {(image_h, image_w)} != mask {(mask_h, mask_w)}."
        )
    print("Dimension check: PASS")

    unique_values = sorted(np.unique(mask_array).tolist())
    print(f"Unique mask values: {unique_values}")

    valid_binary = set(unique_values).issubset({0, 255})
    if not valid_binary:
        raise AssertionError(f"Binary mask check failed: unexpected values {unique_values}.")
    print("Binary mask check: PASS")

    lesion_pixels = int(np.sum(mask_array == 255))
    total_pixels = int(mask_array.size)
    lesion_fraction = lesion_pixels / total_pixels if total_pixels else 0.0

    if lesion_pixels == 0:
        raise AssertionError("Non-empty lesion check failed: mask contains no lesion pixels.")
    if lesion_pixels == total_pixels:
        raise AssertionError("Non-empty lesion check failed: mask is entirely foreground.")
    print(f"Lesion pixel count: {lesion_pixels}")
    print(f"Lesion percentage: {lesion_fraction:.4%}")

    img_min = float(image_array.min())
    img_max = float(image_array.max())
    img_mean = float(image_array.mean())
    print(f"Image stats: min={img_min:.2f}, max={img_max:.2f}, mean={img_mean:.2f}")

    print()
    print("TASK 1 IMAGE / MASK INSPECTION: PASS")


if __name__ == "__main__":
    main()
