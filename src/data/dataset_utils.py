"""General utilities for discovering and loading image data.

This module does not assume a specific dataset.
It is intended to be extended after the dataset structure is known.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_image_files(directory: Path) -> list[Path]:
    """Return supported image files in a directory.

    Args:
        directory: Folder to search.

    Returns:
        Sorted list of image file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    image_files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(image_files)


def load_image(image_path: Path) -> Image.Image:
    """Load an image and convert it to RGB.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image in RGB format.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the file cannot be opened as an image.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except Exception as error:
        raise ValueError(
            f"Could not load image: {image_path}"
        ) from error


def main() -> None:
    """Run a basic self-test using the existing processed folder."""

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    image_files = find_image_files(processed_dir)

    print("=== Dataset Utility Test ===")
    print(f"Directory: {processed_dir}")
    print(f"Images found: {len(image_files)}")

    if not image_files:
        raise RuntimeError("No image files found in the processed directory.")

    for image_path in image_files:
        image = load_image(image_path)
        print(f"[OK] {image_path.name} -> {image.size}, {image.mode}")

    print("=== Dataset utility test successful ===")


if __name__ == "__main__":
    main()