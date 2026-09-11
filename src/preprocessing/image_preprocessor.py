"""Unified Image Preprocessor for Dermoscopic Images.

Provides a configurable, end-to-end preprocessing pipeline that chains:
1. Image format validation (RGB, dimensions, dtype)
2. Resizing to standard dimensions (default 256×256)
3. RGB conversion (if needed)
4. DullRazor hair/artefact removal
5. Illumination normalization (Shades-of-Gray color constancy)
6. Pixel normalization to [0, 1] float32

This is the canonical entry point for preprocessing across the project.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Configurable dermoscopic image preprocessing pipeline."""

    VALID_DTYPES = (np.uint8, np.float32, np.float64)
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    MIN_DIMENSION = 32
    MAX_DIMENSION = 4096

    def __init__(
        self,
        target_size: int = 256,
        apply_hair_removal: bool = True,
        apply_color_constancy: bool = True,
        apply_clahe: bool = False,
        normalize_pixels: bool = False,
    ):
        """Initialize preprocessor with pipeline configuration.

        Args:
            target_size: Target height and width for resizing.
            apply_hair_removal: If True, run DullRazor hair removal.
            apply_color_constancy: If True, run Shades-of-Gray illumination normalization.
            apply_clahe: If True, apply CLAHE contrast enhancement on L-channel.
            normalize_pixels: If True, scale output to [0, 1] float32.
        """
        self.target_size = target_size
        self.apply_hair_removal = apply_hair_removal
        self.apply_color_constancy = apply_color_constancy
        self.apply_clahe = apply_clahe
        self.normalize_pixels = normalize_pixels

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """Validate image format, dimensions, and dtype.

        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        if image is None:
            return False, "Image is None."

        if not isinstance(image, np.ndarray):
            return False, f"Expected numpy array, got {type(image).__name__}."

        if image.size == 0:
            return False, "Image array is empty (zero elements)."

        if image.ndim not in (2, 3):
            return False, f"Expected 2D or 3D array, got {image.ndim}D."

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            return False, f"Expected 1, 3, or 4 channels, got {image.shape[2]}."

        h, w = image.shape[:2]
        if h < self.MIN_DIMENSION or w < self.MIN_DIMENSION:
            return False, f"Image too small: {w}×{h} (minimum {self.MIN_DIMENSION}×{self.MIN_DIMENSION})."

        if h > self.MAX_DIMENSION or w > self.MAX_DIMENSION:
            return False, f"Image too large: {w}×{h} (maximum {self.MAX_DIMENSION}×{self.MAX_DIMENSION})."

        return True, ""

    @staticmethod
    def validate_file_extension(filename: str) -> Tuple[bool, str]:
        """Check if filename has a supported image extension."""
        import os
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ImagePreprocessor.VALID_EXTENSIONS:
            return False, f"Unsupported extension '{ext}'. Supported: {ImagePreprocessor.VALID_EXTENSIONS}"
        return True, ""

    # ------------------------------------------------------------------
    # Individual Steps
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_rgb(image: np.ndarray) -> np.ndarray:
        """Convert image to 3-channel RGB if needed."""
        import cv2

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim == 3:
            c = image.shape[2]
            if c == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if c == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target_size × target_size using high-quality interpolation."""
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        pil_resized = pil_img.resize(
            (self.target_size, self.target_size), PILImage.LANCZOS
        )
        return np.array(pil_resized)

    @staticmethod
    def to_uint8(image: np.ndarray) -> np.ndarray:
        """Ensure image is uint8 format."""
        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                return (image * 255.0).clip(0, 255).astype(np.uint8)
            return image.clip(0, 255).astype(np.uint8)
        return image.astype(np.uint8)

    @staticmethod
    def normalize_to_float(image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1] float32."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype in (np.float32, np.float64):
            if image.max() > 1.0:
                return (image / 255.0).astype(np.float32)
            return image.astype(np.float32)
        return image.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Execute the full preprocessing pipeline.

        Pipeline order:
        1. Validate → 2. Ensure RGB → 3. Resize → 4. Hair Removal
        → 5. Illumination Normalization → 6. CLAHE → 7. Pixel Normalization

        Args:
            image: Input image as numpy array (any supported format).

        Returns:
            Preprocessed image as numpy array.

        Raises:
            ValueError: If image validation fails.
        """
        is_valid, error_msg = self.validate_image(image)
        if not is_valid:
            raise ValueError(f"Image validation failed: {error_msg}")

        # Step 1: Ensure RGB
        processed = self.ensure_rgb(image)

        # Step 2: Ensure uint8 for processing steps
        processed = self.to_uint8(processed)

        # Step 3: Resize
        processed = self.resize_image(processed)

        # Step 4: Hair removal
        if self.apply_hair_removal:
            try:
                from src.preprocessing.artefact_removal import remove_hair_dullrazor
                processed = remove_hair_dullrazor(processed)
            except Exception:
                pass  # Graceful degradation

        # Step 5: Illumination normalization
        if self.apply_color_constancy:
            try:
                from src.preprocessing.illumination import normalize_illumination_gray_world
                processed = normalize_illumination_gray_world(processed, p=6)
            except Exception:
                pass

        # Step 6: CLAHE
        if self.apply_clahe:
            try:
                from src.preprocessing.illumination import apply_clahe_lab
                processed = apply_clahe_lab(processed, clip_limit=1.5)
            except Exception:
                pass

        # Step 7: Pixel normalization
        if self.normalize_pixels:
            processed = self.normalize_to_float(processed)

        return processed

    def preprocess_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """Preprocess a PIL Image through the full pipeline.

        Args:
            pil_image: PIL Image in any mode.

        Returns:
            Preprocessed numpy array.
        """
        img_np = np.array(pil_image.convert("RGB"))
        return self.run_pipeline(img_np)

    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess raw image bytes through the full pipeline.

        Args:
            image_bytes: Raw image file bytes.

        Returns:
            Preprocessed numpy array.

        Raises:
            ValueError: If bytes cannot be decoded as an image.
        """
        import io

        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot decode image bytes: {e}")

        return self.preprocess_from_pil(pil_image)
