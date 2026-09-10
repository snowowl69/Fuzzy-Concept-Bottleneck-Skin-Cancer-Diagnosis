from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ISICSegmentationDataset(Dataset):
    """PyTorch dataset for ISIC 2018 Task 1 segmentation masks."""

    def __init__(
        self,
        csv_file: str | Path,
        image_size: int = 256,
        transform=None,
    ):
        self.csv_file = Path(csv_file)
        if not self.csv_file.is_absolute():
            self.csv_file = PROJECT_ROOT / self.csv_file
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Segmentation CSV not found: {self.csv_file}")

        self.data = pd.read_csv(self.csv_file)
        required_columns = {"image_id", "image_path", "mask_path"}
        missing = required_columns - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if len(self.data) == 0:
            raise ValueError("Segmentation dataset CSV is empty.")

        self.image_size = int(image_size)

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        image_id = str(row["image_id"])

        image_path = Path(str(row["image_path"]))
        mask_path = Path(str(row["mask_path"]))

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        if not mask_path.is_absolute():
            mask_path = PROJECT_ROOT / mask_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for {image_id}: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_id}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_image = mask_image.resize((self.image_size, self.image_size), Image.NEAREST)

        image_tensor = self.transform(image)
        mask_array = np.asarray(mask_image, dtype=np.float32)
        mask_array = (mask_array > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        if image_tensor.shape != (3, self.image_size, self.image_size):
            raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")
        if mask_tensor.shape != (1, self.image_size, self.image_size):
            raise ValueError(f"Unexpected mask tensor shape: {mask_tensor.shape}")

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": image_id,
        }
