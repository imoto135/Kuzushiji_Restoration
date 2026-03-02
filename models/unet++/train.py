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
# ToTensorV2 は albumentations 変換後に非連続配列を生成することがあるため
# 手動テンソル変換に統一し RuntimeError: Numpy is not available を回避する
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

        img  = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_np  = augmented['image']   # すでに Normalize 済み float32 HWC
            mask_np = augmented['mask']
        else:
            aug = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            out = aug(image=img, mask=mask)
            img_np, mask_np = out['image'], out['mask']

        # torch.from_numpy() は PyTorch-NumPy 連携が壊れている環境で失敗するため
        # torch.tensor() でコピー変換することで numpy interop を完全に回避する
        img_t  = torch.tensor(np.ascontiguousarray(img_np).transpose(2, 0, 1), dtype=torch.float32)
        mask_t = torch.tensor(np.ascontiguousarray(mask_np), dtype=torch.float32).unsqueeze(0)
        return img_t, mask_t

# --- Augmentation (Input Dropout採用) ---
def get_transforms(img_size, mode="train"):
    """
    くずし字マスクセグメンテーション向け Augmentation

    文字画像の性質上、以下の拡張は避ける:
      - HorizontalFlip: 文字の鏡像は別文字に見えるため抑制 (p=0.1)
      - VerticalFlip: 文字が完全に逆になるため廃止
      - RandomRotate90: 縦書き文字が横になり非現実的なため廃止

    採用する拡張:
      - 小角度回転 ±10°: 書字のわずかな傾きを模倣
      - CoarseDropout: 損傷（欠損・マスキング）のシミュレーション

    ※ ToTensorV2 は使用しない (非連続配列 RuntimeError を回避するため)
      → __getitem__ 内で np.ascontiguousarray + torch.from_numpy に統一
    """
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.1),             # 文字の鏡像になるため最低限に抑制
            A.Rotate(limit=10, p=0.3),           # 書字の微小傾きを模倣 (±10°)
            A.CoarseDropout(
                max_holes=1, max_height=32, max_width=32,
                min_holes=1, min_height=16, min_width=16,
                fill_value=0, mask_fill_value=None, p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2 は使わない
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2 は使わない
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
        decoder_attention_type='scse',  # Spatial & Channel SE: 細粒度セグメンテーション向けに有効
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16)
    )
    return model.to(device)

# --- 学習ループ ---
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():          # AMP: float16 で forward
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
        scaler.scale(loss).backward()            # AMP: スケーリングされた backward
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    def iou_score(pred, target, eps=1e-6):
        pred = (torch.sigmoid(pred) > 0.5).float()
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        return (inter + eps) / (union + eps)

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating"):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast():      # AMP: validation も fp16 で高速化
                outputs = model(imgs)
            total_loss += loss_fn(outputs, masks).item()
            total_iou += iou_score(outputs, masks).item() * imgs.size(0)

    avg_loss = total_loss / len(loader)
    avg_iou  = total_iou / len(loader.dataset)
    return avg_loss, avg_iou

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='Stage 1: Mask Segmentation Training')
    parser.add_argument('--data_dir', type=str, default='../../data/full_padded')
    parser.add_argument('--output_dir', type=str, default='experiments/unet++_full_characters')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256,  # 64 → 256: 8GB/64→推定32GB/256, 残り40GBに収まる
                        help='バッチサイズ (GPU 0 は ~40GB 空き, 256 で ~32GB 使用見込み)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--run_name', type=str, default='UNet++_full_characters',
                        help='WandB run name (default: auto-generated by WandB)')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience epochs (default: 10). 0 to disable.')
    parser.add_argument('--save_every', type=int, default=10,
                        help='N epoch ごとに定期チェックポイントを保存 (0 で無効, default: 10)')
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
        wandb.init(project="Kuzushiji_Restoration", name=args.run_name, config=vars(args))

    # Data
    logging.info("Loading Datasets...")
    train_ds = KuzushijiSegmentationDataset(
        os.path.join(args.data_dir, 'lq/train'),
        os.path.join(args.data_dir, 'gt_mask/train'),
        transform=get_transforms(args.img_size, mode="train")
    )
    val_ds = KuzushijiSegmentationDataset(
        os.path.join(args.data_dir, 'lq/val'),
        os.path.join(args.data_dir, 'gt_mask/val'),
        transform=get_transforms(args.img_size, mode="val")
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    # Model & Optimization
    logging.info("Building UNet++ (SE-ResNeXt-50)...")
    model = build_model(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
    scaler    = torch.cuda.amp.GradScaler()          # AMP スケーラー
    
    # Loss: 0.5 Dice + 0.5 BCE
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    def criterion(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

    # Training Loop
    best_iou = 0.0
    early_stop_counter = 0
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        scheduler.step(val_iou)

        if not args.no_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "lr": optimizer.param_groups[0]['lr'],
                "early_stop_counter": early_stop_counter,
            })

        # 定期チェックポイント保存
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Checkpoint saved: {ckpt_path}")

        # ベストモデル保存 / Early Stopping
        if val_iou > best_iou:
            best_iou = val_iou
            early_stop_counter = 0
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"New Best IoU: {best_iou:.4f} → Model saved to {save_path}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement. Early stop counter: {early_stop_counter}/{args.early_stop_patience}")
            if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1} (best IoU: {best_iou:.4f})")
                break

    logging.info(f"Training Completed. Best Val IoU: {best_iou:.4f}")
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()