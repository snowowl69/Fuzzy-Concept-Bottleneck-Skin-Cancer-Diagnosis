from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    from src.segmentation.segmentation_dataset import ISICSegmentationDataset

    dataset = ISICSegmentationDataset(
        csv_file=PROJECT_ROOT / "data" / "processed" / "segmentation_train.csv",
        image_size=256,
    )

    sample = dataset[0]

    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}

    print("SEGMENTATION DATASET TEST: PASS")


if __name__ == "__main__":
    main()
