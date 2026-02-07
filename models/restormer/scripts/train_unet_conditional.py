"""Train a conditional U-Net to segment hiragana glyphs in noisy inputs.

The script expects the following layout under ``dataset_final_hiragana``::

    lq_random/
        train|val|test/*.jpg  (noisy character crops)
    mask_gt/
        train|val|test/*.jpg  (binary masks aligned to the inputs)

Masks are read as single-channel images and binarised. Character labels are
derived from the prefix of each filename (e.g. ``U+3042_...``) and converted to
class IDs via ``class_map.csv``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Default hyperparameters tuned for 128x128 hiragana crops.
DEFAULT_IMAGE_SIZE = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 60
DEFAULT_LR = 1e-4
DEFAULT_EMBED_DIM = 32
DEFAULT_PATIENCE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional U-Net training for hiragana masks")
    parser.add_argument("--data-root", default="dataset_final_hiragana", type=str, help="Root directory containing lq_random and mask_gt")
    parser.add_argument("--class-map", default="class_map.csv", type=str, help="CSV with columns char_unicode,class_id")
    parser.add_argument("--encoder", default="resnet34", type=str, help="Backbone encoder name for SMP")
    parser.add_argument("--encoder-weights", default="imagenet", type=str, help="Encoder weights for SMP (pass None for scratch)")
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--embed-dim", default=DEFAULT_EMBED_DIM, type=int, help="Dimensionality of character embedding")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--patience", default=DEFAULT_PATIENCE, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--output-dir", default="experiments/unet_conditional", type=str)
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--run-name", default=None, type=str, help="Optional subdirectory under output-dir")
    parser.add_argument("--predict-split", default="test", type=str, choices=["train", "val", "test"], help="Dataset split to run inference on once training finishes")
    parser.add_argument("--save-predictions", default=None, type=str, help="Directory to store predicted masks; skipped if omitted")
    return parser.parse_args()


def setup_logging(log_dir: Path, level: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_map(csv_path: Path) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    if not {"char_unicode", "class_id"}.issubset(df.columns):
        raise ValueError("class_map.csv must contain char_unicode and class_id columns")
    char_to_id = pd.Series(df.class_id.values, index=df.char_unicode).to_dict()
    return char_to_id


class HiraganaDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, char_to_id: Dict[str, int], transform: A.Compose) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.char_to_id = char_to_id
        self.transform = transform

        image_names = {p.name for p in images_dir.glob("*.jpg")}
        mask_names = {p.name for p in masks_dir.glob("*.jpg")}
        shared = sorted(image_names & mask_names)
        if not shared:
            raise FileNotFoundError(f"No matching jpg files between {images_dir} and {masks_dir}")
        self.filenames = shared

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        filename = self.filenames[index]
        image = Image.open(self.images_dir / filename).convert("RGB")
        mask = Image.open(self.masks_dir / filename).convert("L")

        image_np = np.array(image, dtype=np.uint8)
        mask_np = (np.array(mask, dtype=np.float32) > 127.5).astype(np.float32)

        char_token = filename.split("_")[0]
        if char_token not in self.char_to_id:
            raise KeyError(f"Character token {char_token} missing from class_map")
        class_id = self.char_to_id[char_token]

        augmented = self.transform(image=image_np, mask=mask_np)
        image_tensor = augmented["image"][..., :]
        mask_tensor = augmented["mask"].unsqueeze(0)
        class_tensor = torch.tensor(class_id, dtype=torch.long)
        return image_tensor, mask_tensor, class_tensor, filename


def build_transforms(image_size: int) -> Dict[str, A.Compose]:
    normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-12, 12), p=0.6),
            A.ColorJitter(p=0.3, brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
            normalize,
            ToTensorV2(),
        ]
    )
    eval_transform = A.Compose([A.Resize(image_size, image_size), normalize, ToTensorV2()])
    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


@dataclass
class TrainArtifacts:
    model: nn.Module
    embedding: nn.Embedding
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau


def build_model(num_classes: int, embed_dim: int, encoder: str, encoder_weights: str | None, lr: float, device: torch.device) -> TrainArtifacts:
    embedding = nn.Embedding(num_classes, embed_dim)
    in_channels = 3 + embed_dim
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None if encoder_weights in {"none", "None", ""} else encoder_weights,
        in_channels=in_channels,
        classes=1,
    )
    model = model.to(device)
    embedding = embedding.to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(embedding.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=3)
    return TrainArtifacts(model=model, embedding=embedding, optimizer=optimizer, scheduler=scheduler)


def mix_inputs(images: torch.Tensor, embedding_layer: nn.Embedding, class_ids: torch.Tensor) -> torch.Tensor:
    embedding = embedding_layer(class_ids)
    spatial_shape = images.shape[-2:]
    embedding_map = embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *spatial_shape)
    return torch.cat([images, embedding_map], dim=1)


def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_bin = (preds > 0.5).float()
    intersection = (preds_bin * targets).sum().item()
    union = preds_bin.sum().item() + targets.sum().item() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def train_one_epoch(artifacts: TrainArtifacts, loader: DataLoader, loss_fn, device: torch.device) -> float:
    artifacts.model.train()
    artifacts.embedding.train()
    running_loss = 0.0
    for images, masks, class_ids, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        class_ids = class_ids.to(device)

        inputs = mix_inputs(images, artifacts.embedding, class_ids)

        artifacts.optimizer.zero_grad()
        logits = artifacts.model(inputs)
        loss = loss_fn(logits, masks)
        loss.backward()
        artifacts.optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(artifacts: TrainArtifacts, loader: DataLoader, loss_fn, device: torch.device) -> Tuple[float, float]:
    artifacts.model.eval()
    artifacts.embedding.eval()
    loss_sum = 0.0
    iou_sum = 0.0
    for images, masks, class_ids, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        class_ids = class_ids.to(device)

        inputs = mix_inputs(images, artifacts.embedding, class_ids)
        logits = artifacts.model(inputs)
        loss = loss_fn(logits, masks)
        probs = torch.sigmoid(logits)
        iou_sum += compute_iou(probs, masks) * images.size(0)
        loss_sum += loss.item() * images.size(0)
    return loss_sum / len(loader.dataset), iou_sum / len(loader.dataset)


@torch.no_grad()
def save_predictions(
    checkpoint_path: Path,
    char_to_id: Dict[str, int],
    data_root: Path,
    split: str,
    transforms: Dict[str, A.Compose],
    output_dir: Path,
    device: torch.device,
) -> None:
    if not checkpoint_path.exists():
        logging.warning("Checkpoint %s does not exist. Skipping prediction export.", checkpoint_path)
        return

    logging.info("Loading checkpoint from %s", checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device)
    char_to_id = payload.get("char_to_id", char_to_id)
    embed_dim = payload["config"]["embed_dim"]
    encoder = payload["config"]["encoder"]
    encoder_weights = payload["config"]["encoder_weights"]

    artifacts = build_model(len(char_to_id), embed_dim, encoder, encoder_weights, payload["config"]["lr"], device)
    artifacts.model.load_state_dict(payload["model"])
    artifacts.embedding.load_state_dict(payload["embedding"])
    artifacts.model.eval()
    artifacts.embedding.eval()

    transform_key = "test" if split == "test" else "val"
    dataset = HiraganaDataset(
        data_root / "lq_random" / split,
        data_root / "mask_gt" / split,
        char_to_id,
        transforms[transform_key],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    for image, _, class_id, file_name in tqdm(loader, desc="predict", leave=False):
        if isinstance(file_name, list):
            file_name = file_name[0]
        image = image.to(device, dtype=torch.float32)
        class_id = class_id.to(device)
        inputs = mix_inputs(image, artifacts.embedding, class_id)
        probs = torch.sigmoid(artifacts.model(inputs)).squeeze(0).squeeze(0)
        mask_np = (probs.cpu().numpy() > 0.5).astype(np.uint8) * 255
        Image.fromarray(mask_np, mode="L").save(output_dir / file_name)

    logging.info("Saved predictions for %d images to %s", len(dataset), output_dir)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    set_seeds(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_dir = Path(args.output_dir)
    if args.run_name:
        run_dir = run_dir / args.run_name
    setup_logging(run_dir, args.log_level)

    logging.info("Loading class map from %s", args.class_map)
    char_to_id = load_class_map(Path(args.class_map))
    num_classes = len(char_to_id)
    logging.info("Detected %d unique classes", num_classes)

    transforms = build_transforms(args.image_size)
    data_root = Path(args.data_root)
    datasets = {
        split: HiraganaDataset(
            data_root / "lq_random" / split,
            data_root / "mask_gt" / split,
            char_to_id,
            transforms["train" if split == "train" else "val"],
        )
        for split in ["train", "val"]
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for split in ["train", "val"]
    }

    logging.info("Train samples: %d, Val samples: %d", len(datasets["train"]), len(datasets["val"]))

    artifacts = build_model(num_classes, args.embed_dim, args.encoder, args.encoder_weights, args.lr, device)
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary")

    def loss_fn(pred, target):
        return 0.5 * bce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    best_iou = float("-inf")
    patience_counter = 0
    best_model_path = run_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)
        train_loss = train_one_epoch(artifacts, loaders["train"], loss_fn, device)
        val_loss, val_iou = evaluate(artifacts, loaders["val"], loss_fn, device)

        artifacts.scheduler.step(val_iou)
        logging.info("train_loss=%.4f val_loss=%.4f val_iou=%.4f", train_loss, val_loss, val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save({
                "model": artifacts.model.state_dict(),
                "embedding": artifacts.embedding.state_dict(),
                "char_to_id": char_to_id,
                "config": vars(args),
                "best_iou": best_iou,
            }, best_model_path)
            logging.info("Saved new best model to %s (IoU=%.4f)", best_model_path, best_iou)
        else:
            patience_counter += 1
            logging.info("No improvement. patience %d/%d", patience_counter, args.patience)

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered at epoch %d", epoch)
            break

    logging.info("Training finished. Best IoU %.4f", best_iou)

    if args.save_predictions:
        predictions_dir = Path(args.save_predictions)
        save_predictions(
            best_model_path,
            char_to_id,
            data_root,
            args.predict_split,
            transforms,
            predictions_dir,
            device,
        )


if __name__ == "__main__":
    main()
