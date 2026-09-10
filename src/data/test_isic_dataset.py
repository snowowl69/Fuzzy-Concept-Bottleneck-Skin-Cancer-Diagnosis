from pathlib import Path

import torch

from isic_dataset import ISIC2018Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VALIDATION_CSV = PROCESSED_DIR / "validation.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"


def test_dataset(
    csv_file: Path,
    expected_minimum_size: int,
) -> None:
    print()
    print("=" * 60)
    print(f"TESTING: {csv_file.name}")
    print("=" * 60)

    dataset = ISIC2018Dataset(
        csv_file=csv_file,
        image_size=224,
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) < expected_minimum_size:
        raise AssertionError(
            f"Dataset is unexpectedly small: {len(dataset)}"
        )

    sample = dataset[0]

    image = sample["image"]
    label = sample["label"]
    image_id = sample["image_id"]

    print(f"Image ID: {image_id}")
    print(f"Image tensor shape: {image.shape}")
    print(f"Image tensor dtype: {image.dtype}")
    print(f"Image minimum: {image.min().item():.4f}")
    print(f"Image maximum: {image.max().item():.4f}")
    print(f"Label: {label.item()}")
    print(f"Label dtype: {label.dtype}")

    # ---------------------------------------------------------
    # Validate image
    # ---------------------------------------------------------

    if not isinstance(image, torch.Tensor):
        raise AssertionError("Image is not a PyTorch tensor.")

    if image.shape != (3, 224, 224):
        raise AssertionError(
            f"Unexpected image shape: {image.shape}"
        )

    if image.dtype != torch.float32:
        raise AssertionError(
            f"Unexpected image dtype: {image.dtype}"
        )

    if image.min() < 0 or image.max() > 1:
        raise AssertionError(
            "Image tensor values are outside [0, 1]."
        )

    # ---------------------------------------------------------
    # Validate label
    # ---------------------------------------------------------

    if not isinstance(label, torch.Tensor):
        raise AssertionError("Label is not a PyTorch tensor.")

    if label.dtype != torch.long:
        raise AssertionError(
            f"Unexpected label dtype: {label.dtype}"
        )

    if not 0 <= label.item() <= 6:
        raise AssertionError(
            f"Invalid label index: {label.item()}"
        )

    # ---------------------------------------------------------
    # Validate image ID
    # ---------------------------------------------------------

    if not image_id.startswith("ISIC_"):
        raise AssertionError(
            f"Unexpected image ID: {image_id}"
        )

    print("Result: PASS")


def main() -> None:
    print("=" * 60)
    print("ISIC 2018 PYTORCH DATASET LOADER TEST")
    print("=" * 60)

    test_dataset(
        TRAIN_CSV,
        expected_minimum_size=7000,
    )

    test_dataset(
        VALIDATION_CSV,
        expected_minimum_size=1000,
    )

    test_dataset(
        TEST_CSV,
        expected_minimum_size=1000,
    )

    print()
    print("=" * 60)
    print("PYTORCH DATASET LOADER TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()