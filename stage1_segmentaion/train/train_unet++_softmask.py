#!/usr/bin/env python3
import os
import logging
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pandas as pd
import wandb
import re
import ssl

# SSL設定
ssl._create_default_https_context = ssl._create_unverified_context

# --- ハイパーパラメータ ---
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE = 128
PATIENCE = 10
# --- ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            logging.error(f"Data dirs not found: {image_dir}, {mask_dir}")
            self.pairs = []
            return

        prefix_pattern = re.compile(r"^type\d+_sev[\d\.]+_")
        suffix_pattern = re.compile(r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain)$")

        def get_clean_stem(filename):
            stem = os.path.splitext(filename)[0]
            stem = prefix_pattern.sub("", stem)
            stem = suffix_pattern.sub("", stem)
            return stem

        def build_map(d):
            m = {}
            for f in os.listdir(d):
                if f.startswith("."): continue
                stem = get_clean_stem(f)
                m.setdefault(stem, []).append(f)
            return m

        img_map = build_map(image_dir)
        mask_map = build_map(mask_dir)

        pairs = []
        common_stems = sorted(set(img_map.keys()) & set(mask_map.keys()))
        
        for stem in common_stems:
            img_files = img_map[stem]
            mask_files = mask_map[stem]
            mask_file = mask_files[0] 
            for img_file in img_files:
                pairs.append((img_file, mask_file))

        if len(pairs) == 0:
            logging.error(f"No matching files found between {image_dir} and {mask_dir}")
        else:
            logging.info(f"Dataset created: {len(pairs)} pairs found.")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_fname, mask_fname = self.pairs[idx]
        img = np.array(Image.open(os.path.join(self.image_dir, img_fname)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, mask_fname)).convert("L"), dtype=np.float32)
        mask = mask / 255.0

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        else:
            img = (img.astype(np.float32) / 255.0).transpose(2,0,1)
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask

def get_transforms(img_size, use_aug_dropout=False):
    train_aug_list = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
    ]

    if use_aug_dropout:
        logging.info("Applying CoarseDropout augmentation.")
        train_aug_list.append(
            A.CoarseDropout(
                max_holes=1, max_height=32, max_width=32,
                min_holes=1, min_height=16, min_width=16,
                fill_value=0, mask_fill_value=None, p=0.5
            )
        )

    train_aug_list.extend([
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    train = A.Compose(train_aug_list)

    val = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    return train, val

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Unet++ with Ablations')
    parser.add_argument('--data-dir', type=str, default='datasets/hiragana_dataset')
    parser.add_argument('--train-img', type=str, default='lq/train')
    parser.add_argument('--train-mask', type=str, default='gt_mask/train')
    parser.add_argument('--val-img', type=str, default='lq/val')
    parser.add_argument('--val-mask', type=str, default='gt_mask/val')

    parser.add_argument('--model', type=str, default='unet++', choices=['unet','unet++','attention_unet'])
    parser.add_argument('--encoder-name', type=str, default='se_resnext50_32x4d')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=128)

    parser.add_argument('--save-path', type=str, default='Kuzushiji_Restoration/experiments/segmentation/last/best.pth')
    parser.add_argument('--log-path', type=str, default='Kuzushiji_Restoration/experiments/segmentation/last/train.log')

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--pretrained-encoder', type=str, default='imagenet')

    parser.add_argument('--wandb-project', type=str, default='unet++-ablation')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')

    # --- ここを「深さ5・head-dropoutなし・拡張dropoutあり」用に設定 ---
    # Python 3.8 互換: BooleanOptionalAction を使わずに ON/OFF を提供
    parser.add_argument(
        "--aug-dropout",
        dest="aug_dropout",
        action="store_true",
        default=True,
        help="Use CoarseDropout augmentation (default: enabled)."
    )
    parser.add_argument(
        "--no-aug-dropout",
        dest="aug_dropout",
        action="store_false",
        help="Disable CoarseDropout augmentation."
    )

    # scSEは必要なければOFFのまま
    parser.add_argument('--use-scse', action='store_true', help='Use scSE attention in decoder')

    # 深さは 5 をデフォルトに（=深さ5で学習）
    parser.add_argument('--encoder-depth', type=int, default=5, help='Encoder depth (default: 5)')

    return parser.parse_args()

def format_name(label: str) -> str:
    return ''.join(part.capitalize() for part in re.split("[_\\-]", label) if part)

def format_encoder_name(enc: str) -> str:
    parts = re.split("[_\\-]", enc)
    formatted = []
    for part in parts:
        lower = part.lower()
        if lower.startswith("resnet"):
            formatted.append("ResNet" + part[len("resnet"):])
        elif lower.startswith("se_resnext"):
            formatted.append("SeResNeXt")
        else:
            formatted.append(part.capitalize())
    return "".join(formatted)

def main():
    args = parse_args()
    
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.log_path, mode='w'), logging.StreamHandler()])
    logging.info(f"Start training with {args.encoder_name}")
    logging.info("Parameters: %s", vars(args))
    
    use_wandb = not args.no_wandb
    
    # Run名の自動生成
    extra_tags = []
    if args.aug_dropout:
        extra_tags.append("AugDropout")
    if args.use_scse:
        extra_tags.append("scSE")
    if args.encoder_depth != 5:
        extra_tags.append(f"Depth{args.encoder_depth}")

    tag_str = "_" + "_".join(extra_tags) if extra_tags else "_Baseline"
    run_name = f"{format_name(args.model)}_{format_encoder_name(args.encoder_name)}{tag_str}"

    if use_wandb:
        wandb.init(project="Kuzushiji_Restoration",
                   entity=args.wandb_entity,
                   name=run_name,
                   config=vars(args))

    train_t, val_t = get_transforms(args.image_size, use_aug_dropout=args.aug_dropout)
    
    train_ds = SimpleSegmentationDataset(os.path.join(args.data_dir, args.train_img),
                                        os.path.join(args.data_dir, args.train_mask),
                                        transform=train_t)
    val_ds = SimpleSegmentationDataset(os.path.join(args.data_dir, args.val_img),
                                      os.path.join(args.data_dir, args.val_mask),
                                      transform=val_t)
    if len(train_ds) == 0:
        logging.error("Train dataset empty.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- モデル構築 ---
    encoder_params = dict(
        encoder_name=args.encoder_name,
        encoder_weights=args.pretrained_encoder or None,
        in_channels=3,
        classes=1,
        encoder_depth=args.encoder_depth
    )

    # 【重要】Depthに応じたデコーダーチャンネル調整
    if args.encoder_depth == 3:
        encoder_params["decoder_channels"] = (256, 128, 64)
    elif args.encoder_depth == 4:
        encoder_params["decoder_channels"] = (256, 128, 64, 32)
    elif args.encoder_depth == 5:
        encoder_params["decoder_channels"] = (256, 128, 64, 32, 16)

    decoder_attention_type = 'scse' if args.use_scse else None

    logging.info(f"Model Config: Depth={args.encoder_depth}, DecoderAttn={decoder_attention_type}")

    if args.model == 'unet':
        model = smp.Unet(decoder_attention_type=decoder_attention_type, **encoder_params)
    elif args.model == 'unet++':
        model = smp.UnetPlusPlus(decoder_attention_type=decoder_attention_type, **encoder_params)
    else:
        model = smp.Unet(decoder_attention_type='scse', **encoder_params)
        
    model = model.to(device)
    
    if use_wandb:
        wandb.watch(model, log='all', log_freq=100)

    # loss and optimizer
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce = nn.BCEWithLogitsLoss()
    def loss_fn(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce(pred, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)

    best_iou = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        total_iou = 0.0
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Val {epoch+1}/{args.epochs}"):
                imgs = imgs.to(device, dtype=torch.float)
                masks = masks.to(device, dtype=torch.float)
                outputs = model(imgs)
                preds = torch.sigmoid(outputs)
                total_iou += iou_score(preds, masks).item() * imgs.size(0)
                total_val_loss += loss_fn(outputs, masks).item()
        avg_val_iou = total_iou / (len(val_loader.dataset) if len(val_loader.dataset)>0 else 1)
        avg_val_loss = total_val_loss / (len(val_loader.dataset) if len(val_loader.dataset)>0 else 1)

        logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val IoU: {avg_val_iou:.4f}")
        scheduler.step(avg_val_iou)
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_iou": avg_val_iou})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_iou = avg_val_iou
            torch.save(model.state_dict(), args.save_path)
            if use_wandb:
                # 【変更】Artifactを使ってモデルを明示的にアップロード・バージョン管理する
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}", 
                    type="model",
                    metadata={
                        "val_loss": best_val_loss, 
                        "val_iou": best_iou,
                        "epoch": epoch + 1
                    }
                )
                artifact.add_file(args.save_path)
                wandb.log_artifact(artifact, aliases=["best", "latest"])
                
                wandb.summary["best_val_loss"] = best_val_loss
                wandb.summary["best_val_iou"] = best_iou
            logging.info(f"Saved best model to {args.save_path} (ValLoss {best_val_loss:.4f}, IoU {best_iou:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"No improvement. Patience {patience_counter}/{args.patience}")
        if patience_counter >= args.patience:
            logging.info("Early stopping")
            break

    logging.info("Training finished.")
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()