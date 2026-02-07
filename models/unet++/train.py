import os
import logging
import argparse
import random
import re
import ssl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import wandb

# --- 再現性のためのシード固定 ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- データセット定義 ---
class KuzushijiSegmentationDataset(Dataset):
    """
    くずし字の損傷画像と正解マスクのペアを読み込むデータセット
    ファイル名の不一致（ノイズパラメータ部分）を吸収してペアリングする
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.pairs = self._match_files()

    def _match_files(self):
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Data directories not found: {self.image_dir}, {self.mask_dir}")

        # ファイル名からノイズ情報などを除去してIDを抽出する正規表現
        prefix_pattern = re.compile(r"^type\d+_sev[\d\.]+_")
        suffix_pattern = re.compile(r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain)$")

        def get_clean_stem(filename):
            stem = filename.stem
            stem = prefix_pattern.sub("", stem)
            stem = suffix_pattern.sub("", stem)
            return stem

        # マップ作成
        img_files = sorted([f for f in self.image_dir.iterdir() if f.suffix in ['.jpg', '.png']])
        mask_files = {get_clean_stem(f): f for f in self.mask_dir.iterdir() if f.suffix in ['.jpg', '.png']}

        pairs = []
        for img_path in img_files:
            stem = get_clean_stem(img_path)
            if stem in mask_files:
                pairs.append((img_path, mask_files[stem]))
        
        logging.info(f"Dataset initialized: Found {len(pairs)} pairs.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0  # 0.0 - 1.0 に正規化

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        else:
            transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            augmented = transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return img, mask

# --- Augmentation (Input Dropout採用) ---
def get_transforms(img_size, mode="train"):
    """
    Input Dropout (CoarseDropout) を含むAugmentation設定
    """
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            # 論文の重要ポイント: Input Dropoutによるロバスト性向上
            A.CoarseDropout(
                max_holes=1, max_height=32, max_width=32, 
                min_holes=1, min_height=16, min_width=16,
                fill_value=0, mask_fill_value=None, p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# --- モデル定義 ---
def build_model(device):
    """
    最終採用モデル: UNet++ with SE-ResNeXt-50
    """
    model = smp.UnetPlusPlus(
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_attention_type=None, # 必要に応じて 'scse'
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16)
    )
    return model.to(device)

# --- 学習ループ ---
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    # IOU計算関数 (BCE logit適用済み想定)
    def iou_score(pred, target, eps=1e-6):
        pred = (torch.sigmoid(pred) > 0.5).float()
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        return (inter + eps) / (union + eps)

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            
            total_loss += loss_fn(outputs, masks).item()
            total_iou += iou_score(outputs, masks).item() * imgs.size(0)
            
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader.dataset)
    return avg_loss, avg_iou

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='Stage 1: Mask Segmentation Training')
    parser.add_argument('--data_dir', type=str, default='dataset_final_hiragana')
    parser.add_argument('--output_dir', type=str, default='experiments/best_model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    # Setup
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{args.output_dir}/train.log", mode='w'),
            logging.StreamHandler()
        ]
    )

    # WandB
    if not args.no_wandb:
        wandb.init(project="Kuzushiji_Restoration_Stage1", config=vars(args))

    # Data
    logging.info("Loading Datasets...")
    train_ds = KuzushijiSegmentationDataset(
        os.path.join(args.data_dir, 'lq5/train'),
        os.path.join(args.data_dir, 'mask_gt/train'),
        transform=get_transforms(args.img_size, mode="train")
    )
    val_ds = KuzushijiSegmentationDataset(
        os.path.join(args.data_dir, 'lq5/val'),
        os.path.join(args.data_dir, 'mask_gt/val'),
        transform=get_transforms(args.img_size, mode="val")
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model & Optimization
    logging.info("Building UNet++ (SE-ResNeXt-50)...")
    model = build_model(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
    
    # Loss: 0.5 Dice + 0.5 BCE
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    def criterion(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

    # Training Loop
    best_iou = 0.0
    
    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        scheduler.step(val_iou)
        
        if not args.no_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou, "lr": optimizer.param_groups[0]['lr']})

        # Save Best
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"New Best IoU! Model saved to {save_path}")

    logging.info("Training Completed.")
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()