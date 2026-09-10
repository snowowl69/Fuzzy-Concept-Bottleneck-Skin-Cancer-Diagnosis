from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]


CLASS_NAMES = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]


class ISIC2018Dataset(Dataset):
    """
    PyTorch dataset for the ISIC 2018 Task 3 diagnosis data.

    Each record contains:
        - image_id
        - image_path
        - label
        - label_index
    """

    def __init__(
        self,
        csv_file: str | Path,
        image_size: int = 224,
        transform=None,
    ):
        self.csv_file = Path(csv_file)

        if not self.csv_file.is_absolute():
            self.csv_file = PROJECT_ROOT / self.csv_file

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_file}"
            )

        self.data = pd.read_csv(self.csv_file)

        required_columns = {
            "image_id",
            "image_path",
            "label",
            "label_index",
        }

        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if len(self.data) == 0:
            raise ValueError("Dataset CSV is empty.")

        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size)
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Load and return one image and its label."""

        row = self.data.iloc[index]

        image_id = str(row["image_id"])

        image_path = Path(str(row["image_path"]))

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found for {image_id}: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)

        label = int(row["label_index"])

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return {
            "image": image_tensor,
            "label": label_tensor,
            "image_id": image_id,
        }