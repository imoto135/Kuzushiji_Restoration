"""
predict_mask_unetpp.py

UNet++ (SE-ResNeXt-50) を使って損傷くずし字画像からキャラクターマスクを推論するスクリプト

使い方:
  conda activate unetpp_env
  cd /home/imoto/Kuzushiji_Restoration

  # train / val / test すべてを一括処理 (デフォルト: ソフトマスク出力):
  python scripts/inference/predict_mask_unetpp.py \
      --input_dir  data/full_padded/lq \
      --output_dir outputs/pred_masks_unetpp \
      --model_path models/unet++/experiments/unet++_full_characters/best_model.pth

  # ハードマスク (0/255) も同時に保存する場合:
  python scripts/inference/predict_mask_unetpp.py \
      --input_dir  data/full_padded/lq \
      --output_dir outputs/pred_masks_unetpp \
      --save_binary

  # 特定のサブセットだけ処理 (例: test のみ):
  python scripts/inference/predict_mask_unetpp.py \
      --input_dir  data/full_padded/lq \
      --output_dir outputs/pred_masks_unetpp \
      --splits test

出力構造:
  output_dir/
    train/  ← input_dir/train/ に対応 (ソフトマスク .png がデフォルト)
    val/    ← input_dir/val/   に対応
    test/   ← input_dir/test/  に対応

  デフォルト → ソフトマスク (0〜255 グレースケール, シグモイド確率)
  --save_binary で ハードマスク (_binary.png, 0 or 255) も保存
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp


# ===================== モデル定義 (train.py と同一設定) =====================
def build_model():
    model = smp.UnetPlusPlus(
        encoder_name="se_resnext50_32x4d",
        encoder_weights=None,               # 推論時は ImageNet 重みは不要
        in_channels=3,
        classes=1,
        decoder_attention_type='scse',
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
    )
    return model


# ===================== 前処理 (train.py と同一設定) =====================
def get_transform(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ===================== 推論関数 =====================
@torch.no_grad()
def predict_single(model, img_np: np.ndarray, transform, device: torch.device,
                   threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    1 枚の画像 (H, W, 3 uint8) に対してマスクを推論する

    Returns:
        binary_mask : np.ndarray (H_orig, W_orig) uint8, 0 or 255
        prob_map    : np.ndarray (H_orig, W_orig) uint8, 0-255 (確率 × 255)
    """
    H_orig, W_orig = img_np.shape[:2]

    # 前処理 (リサイズ + Normalize)
    aug = transform(image=img_np)
    img_t = torch.tensor(
        np.ascontiguousarray(aug['image']).transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0).to(device)  # (1, 3, H, W)

    # AMP で forward
    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        logit = model(img_t)  # (1, 1, H, W)

    prob = torch.sigmoid(logit).squeeze().cpu().float()  # (H, W)  0-1

    # 元の解像度にリサイズ
    prob_resized = F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=(H_orig, W_orig),
        mode='bilinear', align_corners=False
    ).squeeze()

    # .numpy() は unetpp_env で失敗するため、tolist() 経由で numpy 化を回避
    prob_resized_np = np.array(prob_resized.tolist(), dtype=np.float32)
    prob_uint8   = (prob_resized_np * 255).clip(0, 255).astype(np.uint8)
    binary_mask  = (prob_resized_np >= threshold).astype(np.uint8) * 255

    return binary_mask, prob_uint8


# ===================== メイン =====================
def main():
    parser = argparse.ArgumentParser(description='UNet++ によるキャラクターマスク推論')
    parser.add_argument('--input_dir',  type=str, required=True,
                        help='lq ルートディレクトリ (train/val/test サブディレクトリを含む)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='マスク画像の保存先ルートディレクトリ (train/val/test が作られる)')
    parser.add_argument('--model_path', type=str,
                        default='models/unet++/experiments/unet++_full_characters/best_model.pth',
                        help='学習済みモデルのパス')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val', 'test'],
                        help='処理するスプリット (default: train val test)')
    parser.add_argument('--img_size',   type=int, default=128,
                        help='推論時の入力解像度 (学習時と揃える, default: 128)')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='2値化の閾値 (default: 0.5)')
    parser.add_argument('--save_binary', action='store_true',
                        help='ハードマスク (0/255) も _binary.png として保存する (ソフトマスクは常に保存される)')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp'],
                        help='対象とする画像拡張子')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Model  : {args.model_path}")
    print(f"Splits : {args.splits}")

    # モデルロード (一度だけ)
    model = build_model().to(device)
    state = torch.load(args.model_path, map_location=device)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params\n")

    transform = get_transform(args.img_size)
    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    total_saved = 0

    for split in args.splits:
        split_in  = input_root  / split
        split_out = output_root / split

        if not split_in.exists():
            print(f"[SKIP] {split_in} が存在しません")
            continue

        split_out.mkdir(parents=True, exist_ok=True)

        img_paths = sorted([
            p for p in split_in.iterdir()
            if p.suffix.lower() in args.extensions
        ])

        if not img_paths:
            print(f"[SKIP] {split_in}: 画像が見つかりません")
            continue

        print(f"=== {split} : {len(img_paths)} 枚 ===")

        for img_path in tqdm(img_paths, desc=split):
            try:
                img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            except Exception as e:
                print(f"[SKIP] {img_path.name}: {e}")
                continue

            binary_mask, prob_map = predict_single(
                model, img_np, transform, device, args.threshold
            )

            # ソフトマスク保存 (デフォルト: 0〜255 グレースケール 確率マップ)
            out_path = split_out / (img_path.stem + '.png')
            Image.fromarray(prob_map).save(out_path)

            # ハードマスク (0/255) も保存 (オプション)
            if args.save_binary:
                binary_path = split_out / (img_path.stem + '_binary.png')
                Image.fromarray(binary_mask).save(binary_path)

        total_saved += len(img_paths)
        print(f"  → {split_out} に保存完了")

    print(f"\n全処理完了: 合計 {total_saved} 枚")
    print(f"保存先: {output_root}")
    print(f"出力形式: ソフトマスク (0〜255 シグモイド確率)")
    if args.save_binary:
        print(f"ハードマスク (0/255) も _binary.png で保存済み")


if __name__ == '__main__':
    main()
