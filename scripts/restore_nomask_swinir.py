#!/usr/bin/env python3
"""
SwinIR (NoMask) を使用したくずし字画像修復スクリプト
マスクなしで学習した SwinIR モデル (net_g_750000.pth) を使用して損傷画像を修復します。
"""

import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

# nafnet の basicsr を直接インポート
nafnet_path = Path(__file__).parent.parent / "models" / "nafnet"
sys.path.insert(0, str(nafnet_path))

from basicsr.models.archs.SwinIR_arch import SwinIR


def parse_args():
    parser = argparse.ArgumentParser(description="SwinIR NoMask Image Restoration for Kuzushiji")

    parser.add_argument(
        "--weights",
        type=str,
        default="models/nafnet/experiments/SwinIR_Kuzushiji_NoMask_CharbPercep/models/net_g_750000.pth",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/hiragana_fulldataset_5stain/lq/test",
        help="Directory containing damaged images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/swinir_nomask_charbpercep",
        help="Directory to save restored images"
    )

    # wandb設定
    parser.add_argument('--use_wandb', action='store_true', help='wandbに結果を記録する')
    parser.add_argument('--wandb_project', type=str, default='Kuzushiji_Restoration', help='wandbプロジェクト名')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run名')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='wandbタグ')

    return parser.parse_args()


def load_model(weights_path, device):
    """SwinIR (NoMask) モデルをロード"""
    logging.info(f"Loading model from {weights_path}")

    model = SwinIR(
        upscale=1,
        in_chans=3,           # RGB のみ（マスクなし）
        out_chans=3,          # RGB 出力
        img_size=128,
        patch_size=1,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2.0,
        upsampler='',         # '' = denoising mode
        resi_connection='1conv'
    )

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)

    logging.info("Model loaded successfully")
    return model


def process_image(model, img_path, device):
    """単一画像の修復処理（マスクなし）"""

    # 画像読み込み (BGR -> RGB)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 正規化: [0, 255] -> [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # HWC -> CHW -> バッチ次元追加
    img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        output = model(img_tensor)

    # テンソル -> numpy: [0, 1] -> [0, 255]
    output_np = output.squeeze(0).cpu().numpy()
    output_np = np.transpose(output_np, (1, 2, 0))  # CHW -> HWC
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

    # RGB -> BGR (OpenCV用)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

    return output_bgr


def main():
    args = parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # wandb初期化
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            job_type='inference',
            tags=args.wandb_tags,
            config=vars(args)
        )

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # パス設定
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 画像ファイルのリスト取得
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if len(image_files) == 0:
        logging.error(f"No images found in {input_dir}")
        return

    logging.info(f"Found {len(image_files)} images to process")

    # モデルロード
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logging.error(f"Weights not found: {weights_path}")
        return
    model = load_model(weights_path, device)

    # 画像処理
    processed_count = 0
    for img_path in tqdm(image_files, desc="Restoring images"):
        try:
            restored_img = process_image(model, img_path, device)

            output_path = output_dir / f"{img_path.stem}_restored.png"
            success = cv2.imwrite(str(output_path), restored_img)

            if not success:
                logging.error(f"Failed to save {output_path}")
            else:
                processed_count += 1

        except Exception as e:
            logging.error(f"Error processing {img_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    logging.info(f"Restoration completed. {processed_count}/{len(image_files)} images saved to {output_dir}")

    # wandb記録
    if args.use_wandb:
        wandb.log({"inference/processed_images": processed_count})
        wandb.finish()


if __name__ == "__main__":
    main()
