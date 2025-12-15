#!/usr/bin/env python3
import os
import logging
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import wandb
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_paths = sorted([p for p in self.image_dir.iterdir() if p.is_file()])
        self.mask_paths = {p.stem: p for p in self.mask_dir.iterdir() if p.is_file()}
        self.pairs = [(p, self.mask_paths.get(p.stem)) for p in self.image_paths]
        self.pairs = [(img, mask) for img, mask in self.pairs if mask is not None]
        if len(self.pairs) == 0:
            logging.error("SoftMaskDataset: no matching pairs found between %s and %s", image_dir, mask_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


def build_transforms(image_size):
    train = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train, val


def iou_score(preds, targets, eps=1e-6):
    preds = (preds > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, save_path='best_model.pth', maximize=True):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf") if maximize else float("inf")
        self.counter = 0
        self.save_path = save_path
        self.maximize = maximize

    def step(self, metric, model, minimize=False):
        if self.maximize:
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            try:
                torch.save(model.state_dict(), self.save_path)
            except Exception:
                logging.exception("Failed to save model for early stopping.")
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def format_encoder_name(enc: str) -> str:
    parts = re.split("[_\\-]", enc)
    formatted = []
    for part in parts:
        lower = part.lower()
        if lower.startswith("resnet"):
            formatted.append("ResNet" + part[len("resnet"):])
        else:
            formatted.append(part.capitalize())
    return "".join(formatted)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepLabV3+ soft mask training")
    parser.add_argument("--data-dir", type=str, default="Kuzushiji_Restoration/datasets/dataset_final_hiragana")
    parser.add_argument("--train-img", type=str, default="lq_random/train")
    parser.add_argument("--train-mask", type=str, default="mask_gt/train")
    parser.add_argument("--val-img", type=str, default="lq_random/val")
    parser.add_argument("--val-mask", type=str, default="mask_gt/val")
    parser.add_argument("--encoder", type=str, default="efficientnet-b4")
    parser.add_argument("--wandb-project", type=str, default="Kuzushiji_Restoration")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--save-path", type=str, default="Kuzushiji_Restoration/experiments/deeplabv3p_softmask_best.pth")
    parser.add_argument("--log-path", type=str, default="Kuzushiji_Restoration/experiments/deeplabv3p_softmask.log")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-restore-best", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_path, mode="w")])
    logging.info("Parameters: %s", vars(args))

    train_t, val_t = build_transforms(args.image_size)
    train_ds = SoftMaskDataset(os.path.join(args.data_dir, args.train_img),
                               os.path.join(args.data_dir, args.train_mask),
                               transform=train_t)
    val_ds = SoftMaskDataset(os.path.join(args.data_dir, args.val_img),
                             os.path.join(args.data_dir, args.val_mask),
                             transform=val_t)

    if len(train_ds) == 0 or len(val_ds) == 0:
        logging.error("Train/Val dataset empty. Check paths.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    enc_weights = None if str(args.encoder_weights).lower() in ("none", "null") else args.encoder_weights
    model = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights=enc_weights,
                              in_channels=3, classes=1).to(DEVICE)
    use_wandb = not args.no_wandb
    run_name = f"DeepLabV3Plus_{format_encoder_name(args.encoder)}_SoftMask"
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args))
        wandb.watch(model, log="all", log_freq=100)

    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()

    def loss_fn(pred, target):
        return 0.5 * dice(pred, target) + 0.5 * bce(pred, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_loss = float("inf")
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta, save_path=args.save_path, maximize=False)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}"):
            imgs = imgs.to(DEVICE, dtype=torch.float)
            masks = masks.to(DEVICE, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Val {epoch+1}/{args.epochs}"):
                imgs = imgs.to(DEVICE, dtype=torch.float)
                masks = masks.to(DEVICE, dtype=torch.float)
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                val_iou += iou_score(torch.sigmoid(outputs), masks) * imgs.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_iou = val_iou / len(val_loader.dataset)

        logging.info(f"Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, ValIoU={avg_val_iou:.4f}")
        scheduler.step(avg_val_iou)
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_iou": avg_val_iou})
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_path)
            if use_wandb:
                wandb.save(args.save_path, base_path=os.path.dirname(args.save_path))
            logging.info("Model improved and saved.")
        if early_stopper.step(avg_val_loss, model, minimize=True):
            logging.info("Early stopping triggered.")
            break

    logging.info("Training finished.")
    if args.early_stop_restore_best and os.path.isfile(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=DEVICE))
        logging.info("Restored best model after early stopping.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()