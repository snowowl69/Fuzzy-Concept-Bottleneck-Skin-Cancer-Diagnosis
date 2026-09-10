from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    from src.segmentation.segmentation_dataset import ISICSegmentationDataset

    csv_path = PROJECT_ROOT / "data" / "processed" / "segmentation_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected segmentation training CSV at {csv_path}.")

    dataset = ISICSegmentationDataset(csv_file=csv_path, image_size=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    images = batch["image"]
    masks = batch["mask"]
    image_ids = batch["image_id"]

    assert images.shape[0] == 4
    assert images.shape == (4, 3, 256, 256)
    assert masks.shape == (4, 1, 256, 256)
    assert len(image_ids) == 4
    assert torch.isfinite(images).all()
    assert torch.isfinite(masks).all()
    assert torch.all((masks == 0.0) | (masks == 1.0))
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0
    assert len(set(image_ids)) == len(image_ids)

    print("SEGMENTATION DATALOADER TEST: PASS")


if __name__ == "__main__":
    main()
