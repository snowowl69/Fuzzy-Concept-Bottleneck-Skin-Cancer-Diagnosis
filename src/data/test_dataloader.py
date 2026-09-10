from pathlib import Path

import torch
from torch.utils.data import DataLoader

from isic_dataset import ISIC2018Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

BATCH_SIZE = 8


def main() -> None:
    print("=" * 60)
    print("ISIC 2018 DATALOADER TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Create dataset
    # ---------------------------------------------------------

    dataset = ISIC2018Dataset(
        csv_file=TRAIN_CSV,
        image_size=224,
    )

    print(f"Dataset size: {len(dataset)}")

    # ---------------------------------------------------------
    # Create DataLoader
    # ---------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print(f"Batch size: {BATCH_SIZE}")
    print("Shuffle: True")
    print("Workers: 0")

    # ---------------------------------------------------------
    # Load one batch
    # ---------------------------------------------------------

    batch = next(iter(dataloader))

    images = batch["image"]
    labels = batch["label"]
    image_ids = batch["image_id"]

    print()
    print("-" * 60)
    print("FIRST BATCH")
    print("-" * 60)

    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Number of image IDs: {len(image_ids)}")

    print()
    print("Image IDs:")

    for image_id in image_ids:
        print(f"  {image_id}")

    print()
    print("Labels:")
    print(labels.tolist())

    # ---------------------------------------------------------
    # Validate batch
    # ---------------------------------------------------------

    if images.shape != (BATCH_SIZE, 3, 224, 224):
        raise AssertionError(
            f"Unexpected image batch shape: {images.shape}"
        )

    if labels.shape != (BATCH_SIZE,):
        raise AssertionError(
            f"Unexpected label batch shape: {labels.shape}"
        )

    if images.dtype != torch.float32:
        raise AssertionError(
            f"Unexpected image dtype: {images.dtype}"
        )

    if labels.dtype != torch.long:
        raise AssertionError(
            f"Unexpected label dtype: {labels.dtype}"
        )

    if images.min().item() < 0:
        raise AssertionError(
            "Image values below 0 detected."
        )

    if images.max().item() > 1:
        raise AssertionError(
            "Image values above 1 detected."
        )

    if not torch.all(
        (labels >= 0) & (labels <= 6)
    ):
        raise AssertionError(
            "Invalid label index detected."
        )

    if len(image_ids) != BATCH_SIZE:
        raise AssertionError(
            "Unexpected number of image IDs."
        )

    # ---------------------------------------------------------
    # Verify all IDs are unique in the batch
    # ---------------------------------------------------------

    if len(set(image_ids)) != BATCH_SIZE:
        raise AssertionError(
            "Duplicate image IDs detected in batch."
        )

    print()
    print("=" * 60)
    print("DATALOADER TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()