import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import logging
import pandas as pd

# --- 学習パラメータ設定 ---
NUM_EPOCHS = 100 
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE = 128
PATIENCE = 10 
# ★★★ エンベディングの次元数 ★★★
EMBEDDING_DIM = 64
# --- ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ConditionalSegmentationDataset クラスは変更なし
class ConditionalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, char_to_id_map, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.char_to_id = char_to_id_map
        
        image_filenames = set(os.listdir(image_dir))
        mask_filenames = set(os.listdir(mask_dir))
        self.images = sorted(list(image_filenames.intersection(mask_filenames)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 128).astype(np.float32)

        # ファイル名から文字ラベルを取得し、クラスID (整数) を取得
        char_label = filename.split('_')[0]
        class_id = self.char_to_id.get(char_label, 0)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        
        # 画像、マスク、そして「クラスID」の3つを返す
        return image, mask, class_id

def main():
    parser = argparse.ArgumentParser(description="Conditional Segmentation Training with Embedding (DeepLabV3+)")
    # モデル選択の引数は今回は DeepLabV3+ 固定なので削除
    args = parser.parse_args()
    
    # ファイル名を deeplabv3p 用に変更
    MODEL_SAVE_PATH = "embedding_deeplabv3p_best_model.pth"
    LOG_FILE_PATH = "embedding_deeplabv3p_training_random.log"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])
    logging.info(f"--- 条件付きモデル 'DeepLabV3+' の学習を開始します (Embedding) ---")

    try:
        class_map_df = pd.read_csv("class_map.csv")
        char_to_id = pd.Series(class_map_df.class_id.values, index=class_map_df.char_unicode).to_dict()
        num_classes = len(char_to_id)
        logging.info(f"クラスマップをロードしました。クラス総数: {num_classes}")
    except FileNotFoundError:
        logging.error("エラー: class_map.csvが見つかりません。")
        return

    DATA_DIR = "dataset_final_hiragana"
    data_transforms = {
        "train": A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-15, 15), p=0.7), A.ElasticTransform(alpha=1, sigma=30, p=0.3), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
        "val": A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
    }

    train_dataset = ConditionalSegmentationDataset(os.path.join(DATA_DIR, 'lq_random', 'train'), os.path.join(DATA_DIR, 'mask_gt', 'train'), char_to_id, transform=data_transforms["train"])
    val_dataset = ConditionalSegmentationDataset(os.path.join(DATA_DIR, 'lq_random', 'val'), os.path.join(DATA_DIR, 'mask_gt', 'val'), char_to_id, transform=data_transforms["val"])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ★★★ ここが DeepLabV3+ への変更点 ★★★
    # 1. エンベディング層を定義 (変更なし)
    embedding_layer = nn.Embedding(num_classes, EMBEDDING_DIM).to(device)

    # 2. モデルの入力チャンネル数を変更 (3ch + 64ch) (変更なし)
    total_in_channels = 3 + EMBEDDING_DIM
    
    # 3. DeepLabV3Plus モデルを定義
    # DeepLabV3Plus を使用し、in_channelsを total_in_channels に設定
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",    # 元のスクリプトに合わせて efficientnet-b7 を使用
        encoder_weights="imagenet",       
        in_channels=total_in_channels,    # 結合された入力チャンネル数
        classes=1                         # バイナリセグメンテーションなので1
    )
    model = model.to(device)
    
    dice_loss = smp.losses.DiceLoss(mode='binary'); bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    loss_fn = lambda p, t: 0.5 * dice_loss(p, t) + 0.5 * bce_loss(p, t)
    
    # 4. オプティマイザにエンベディング層のパラメータも追加 (変更なし)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(embedding_layer.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)

    best_iou = 0.0; patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train(); embedding_layer.train(); train_loss = 0.0
        # 5. 学習ループの変更 (変更なし: 入力処理はそのまま)
        for images, masks, class_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, masks, class_ids = images.to(device, dtype=torch.float), masks.to(device, dtype=torch.float), class_ids.to(device)
            
            # クラスIDをエンベディングベクトルに変換
            embedding = embedding_layer(class_ids) # (B, 32)
            # エンベディングマップを作成
            embedding_map = embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, IMAGE_SIZE, IMAGE_SIZE) # (B, 32, 128, 128)
            
            # 画像とエンベディングマップを結合
            inputs = torch.cat([images, embedding_map], dim=1)
            
            optimizer.zero_grad(); outputs = model(inputs); loss = loss_fn(outputs, masks); loss.backward(); optimizer.step()
            train_loss += loss.item()

        model.eval(); embedding_layer.eval(); total_intersection, total_union = 0.0, 0.0
        with torch.no_grad():
            # 6. 検証ループの変更 (変更なし: 入力処理はそのまま)
            for images, masks, class_ids in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, masks, class_ids = images.to(device, dtype=torch.float), masks.to(device, dtype=torch.float), class_ids.to(device)

                embedding = embedding_layer(class_ids)
                embedding_map = embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, IMAGE_SIZE, IMAGE_SIZE)
                inputs = torch.cat([images, embedding_map], dim=1)

                outputs = model(inputs); preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum().item()
                union = preds.sum().item() + masks.sum().item() - intersection
                total_intersection += intersection
                total_union += union
        
        avg_val_iou = (total_intersection + 1e-6) / (total_union + 1e-6)
        avg_train_loss = train_loss / len(train_loader)
        
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val IoU: {avg_val_iou:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(avg_val_iou)

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou; torch.save(model.state_dict(), MODEL_SAVE_PATH); logging.info(f"-> Best model saved to {MODEL_SAVE_PATH} (IoU: {best_iou:.4f})"); patience_counter = 0
        else:
            patience_counter += 1; logging.info(f"IoU did not improve. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE: logging.info(f"\nEarly stopping triggered."); break

    logging.info(f"\nモデル 'DeepLabV3+' の学習が完了しました。")

if __name__ == '__main__':
    main()