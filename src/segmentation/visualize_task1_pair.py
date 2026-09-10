from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "isic2018" / "task1_input" / "ISIC2018_Task1-2_Training_Input"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "isic2018" / "task1_groundtruth" / "ISIC2018_Task1_Training_GroundTruth"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "segmentation_checks"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPEG files found under {IMAGE_DIR}.")

    image_path = image_files[0]
    mask_path = next(MASK_DIR.glob(f"{image_path.stem}_segmentation.png"), None)
    if mask_path is None:
        raise FileNotFoundError(f"No matching mask for {image_path.name}.")

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    original_out = OUTPUT_DIR / f"{image_path.stem}_original.png"
    mask_out = OUTPUT_DIR / f"{image_path.stem}_mask_groundtruth.png"
    overlay_out = OUTPUT_DIR / f"{image_path.stem}_overlay.png"

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground-truth mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    overlay = image.copy()
    mask_array = mask.point(lambda p: 255 if p > 0 else 0)
    overlay = overlay.convert("RGBA")
    red = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red.putalpha(mask_array)
    overlay = Image.alpha_composite(overlay, red)
    plt.imshow(overlay)
    plt.title("Image + mask overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(overlay_out, dpi=150)
    plt.close()

    image.save(original_out)
    mask.save(mask_out)

    for path in [original_out, mask_out, overlay_out]:
        if not path.exists():
            raise FileNotFoundError(f"Expected output file was not created: {path}")

    print(f"Image ID: {image_path.stem}")
    print(f"Image path: {image_path}")
    print(f"Mask path: {mask_path}")
    print(f"Original output path: {original_out}")
    print(f"Mask output path: {mask_out}")
    print(f"Overlay output path: {overlay_out}")


if __name__ == "__main__":
    main()
