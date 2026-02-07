#!/usr/bin/env python3
"""
卒論 第1段階: 損傷領域推定（ソフトマスク生成）推論スクリプト
学習済みモデルを用いて画像から損傷箇所の確率マップ（ソフトマスク）を生成します。
生成されたマスクは第2段階の修復モデル（Restormer）の入力として使用されます。
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- 設定 ---
# 卒論のベストモデル構成に合わせる
MODEL_ARCH = "unet++"
ENCODER_NAME = "se_resnext50_32x4d"

class InferenceDataset(Dataset):
    """
    推論用データセット
    画像のリサイズ処理を行うが、後で復元するために「元のサイズ」も返す
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        # 画像ファイルのみを対象にする
        self.paths = sorted([
            p for p in self.image_dir.iterdir() 
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # 画像読み込み
        img_pil = Image.open(path).convert("RGB")
        w, h = img_pil.size
        img_arr = np.array(img_pil)

        # 前処理（リサイズ・正規化）
        if self.transform:
            augmented = self.transform(image=img_arr)
            img_tensor = augmented['image']
        else:
            # transformがない場合のフォールバック（通常は使用しない）
            transform = A.Compose([A.ToTensorV2()])
            img_tensor = transform(image=img_arr)['image']

        # ファイル名, Tensor画像, 元の高さ, 元の幅 を返す
        return path.name, img_tensor, h, w

def get_transforms(img_size):
    """学習時と同じ正規化・リサイズ処理"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def build_model(device, checkpoint_path=None):
    """モデル構築と重みのロード"""
    # モデル定義 (学習コードと一致させる)
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None, # 推論時は学習済み重みをロードするためNoneで初期化
        in_channels=3,
        classes=1,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16)
    )

    if checkpoint_path:
        logging.info(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    
    return model.to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Soft Mask Generation")
    
    # 必須引数
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated masks")
    
    # オプション（デフォルト値は論文の設定に準拠）
    parser.add_argument("--image-size", type=int, default=128, help="Input size for the model (default: 128)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ロギング設定
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # データセット・データローダー
    transform = get_transforms(args.image_size)
    dataset = InferenceDataset(args.input_dir, transform=transform)
    
    if len(dataset) == 0:
        logging.error(f"No images found in {args.input_dir}")
        return

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # モデル準備
    model = build_model(device, args.weights)
    model.eval()

    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Start inference on {len(dataset)} images...")

    with torch.no_grad():
        for filenames, imgs, orig_hs, orig_ws in tqdm(loader, desc="Processing"):
            imgs = imgs.to(device, dtype=torch.float)
            
            # 推論実行
            outputs = model(imgs)
            probs = torch.sigmoid(outputs) # 0.0-1.0の確率値に変換

            # バッチ内の各画像について処理
            for i, name in enumerate(filenames):
                prob_map = probs[i].squeeze(0) # (1, H, W) -> (H, W)
                
                # 元の画像サイズに戻す (Bilinear補間)
                # prob_map は GPU 上の Tensor なので、unsqueeze で次元を足して interpolate する
                orig_h = orig_hs[i].item()
                orig_w = orig_ws[i].item()
                
                prob_map_resized = F.interpolate(
                    prob_map.unsqueeze(0).unsqueeze(0), 
                    size=(orig_h, orig_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()

                # NumPy化して保存 (0-255のグレースケール画像)
                mask_np = (prob_map_resized.cpu().numpy() * 255).astype(np.uint8)
                
                # 保存パス
                # 入力が jpg でも png でも、マスクは可逆圧縮の PNG か、高画質 JPEG で保存推奨
                out_path = Path(args.output_dir) / Path(name).with_suffix('.png') # PNGで保存
                Image.fromarray(mask_np).save(out_path)

    logging.info(f"Inference finished. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()