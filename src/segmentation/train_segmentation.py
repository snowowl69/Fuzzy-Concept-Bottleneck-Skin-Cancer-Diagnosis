from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation_dataset import ISICSegmentationDataset
from unet_baseline import UNetBaseline
from segmentation_losses import bce_dice_loss
from segmentation_metrics import dice_coefficient, iou_score


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "segmentation_train.csv"
)

VALIDATION_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "segmentation_validation.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

CHECKPOINT_PATH = (
    MODEL_DIR / "segmentation_unet_baseline.pt"
)

HISTORY_PATH = (
    REPORT_DIR / "segmentation_training_history.csv"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 256

BATCH_SIZE = 1

EPOCHS = 5

LEARNING_RATE = 1e-3

NUM_WORKERS = 0

CPU_THREADS = 2

DEVICE = torch.device("cpu")
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(CPU_THREADS)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds where practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# DATA LOADING
# ============================================================

def create_dataloaders():
    """Create training and validation datasets/loaders."""

    train_dataset = ISICSegmentationDataset(
        csv_file=TRAIN_CSV,
        image_size=IMAGE_SIZE,
    )

    validation_dataset = ISICSegmentationDataset(
        csv_file=VALIDATION_CSV,
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model: torch.nn.Module,
    validation_loader: DataLoader,
):
    """Evaluate model on the validation set."""

    model.eval()

    total_loss = 0.0
    dice_values = []
    iou_values = []

    with torch.no_grad():

        for batch in validation_loader:

            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            logits = model(images)

            loss = bce_dice_loss(
                logits,
                masks,
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            predictions_np = (
                predictions
                .cpu()
                .numpy()
            )

            masks_np = (
                masks
                .cpu()
                .numpy()
            )

            for prediction, target in zip(
                predictions_np,
                masks_np,
            ):
                dice_values.append(
                    dice_coefficient(
                        prediction,
                        target,
                    )
                )

                iou_values.append(
                    iou_score(
                        prediction,
                        target,
                    )
                )

    dataset_size = len(validation_loader.dataset)

    average_loss = (
        total_loss / dataset_size
    )

    average_dice = float(
        np.mean(dice_values)
    )

    average_iou = float(
        np.mean(iou_values)
    )

    return (
        average_loss,
        average_dice,
        average_iou,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_dice: float,
    validation_iou: float,
) -> None:
    """Save the best segmentation checkpoint."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "validation_dice": validation_dice,
        "validation_iou": validation_iou,
        "config": {
            "seed": SEED,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "num_workers": NUM_WORKERS,
            "device": str(DEVICE),
            "model": "UNetBaseline",
            "base_channels": 32,
        },
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )


# ============================================================
# TRAINING
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ISIC 2018 TASK 1 SEGMENTATION TRAINING")
    print("=" * 70)

    print()
    print("Configuration")
    print("-" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Validation CSV: {VALIDATION_CSV}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Random seed: {SEED}")

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV}"
        )

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {VALIDATION_CSV}"
        )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    ) = create_dataloaders()

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    print(
        f"Training batches per epoch: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating model...")

    model = UNetBaseline(
        in_channels=3,
        out_channels=1,
        base_channels=32,
    )

    model = model.to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_validation_dice = -1.0
    best_validation_iou = 0.0

    history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    overall_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.perf_counter()

        model.train()

        total_training_loss = 0.0

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):

            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = bce_dice_loss(
                logits,
                masks,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected."
                )

            loss.backward()

            optimizer.step()

            total_training_loss += (
                loss.item()
                * images.size(0)
            )

            if batch_index == 1:
                print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"- first batch loss: "
                    f"{loss.item():.6f}"
                )

        train_size = len(
            train_loader.dataset
        )

        training_loss = (
            total_training_loss
            / train_size
        )

        (
            validation_loss,
            validation_dice,
            validation_iou,
        ) = validate(
            model,
            validation_loader,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print(
            f"Train loss: {training_loss:.6f}"
        )
        print(
            f"Val loss:   {validation_loss:.6f}"
        )
        print(
            f"Val Dice:   {validation_dice:.6f}"
        )
        print(
            f"Val IoU:    {validation_iou:.6f}"
        )
        print(
            f"Epoch time: {epoch_time:.2f} seconds"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "val_loss": validation_loss,
                "val_dice": validation_dice,
                "val_iou": validation_iou,
            }
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if validation_dice > best_validation_dice:

            best_validation_dice = validation_dice
            best_validation_iou = validation_iou

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_dice=validation_dice,
                validation_iou=validation_iou,
            )

            print(
                "Best checkpoint updated."
            )

    total_training_time = (
        time.perf_counter()
        - overall_start
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice",
                "val_iou",
            ],
        )

        writer.writeheader()

        writer.writerows(history)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.6f}"
    )

    print(
        f"Best validation IoU: "
        f"{best_validation_iou:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Training history: "
        f"{HISTORY_PATH}"
    )

    print(
        f"Total training time: "
        f"{total_training_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()