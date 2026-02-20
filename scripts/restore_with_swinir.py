#!/usr/bin/env python3
"""
SwinIRを使用したくずし字画像修復スクリプト
学習済みモデル (net_g_195000.pth) を使用して損傷画像を修復します。
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
    parser = argparse.ArgumentParser(description="SwinIR Image Restoration for Kuzushiji")
    
    parser.add_argument(
        "--weights",
        type=str,
        default="models/swinir/experiments/SwinIR_PredMask_CharbPercep/models/net_g_200000.pth",
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
        default="outputs/swinir_restoration_charbpercep",
        help="Directory to save restored images"
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="data/hiragana_fulldataset_5stain/pred_mask/test",
        help="Directory containing damage masks"
    )
    
    # wandb設定
    parser.add_argument('--use_wandb', action='store_true', help='wandbに結果を記録する')
    parser.add_argument('--wandb_project', type=str, default='Kuzushiji_Restoration', help='wandbプロジェクト名')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run名')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='wandbタグ')
    
    return parser.parse_args()


def load_model(weights_path, device):
    """SwinIRモデルをロード"""
    logging.info(f"Loading model from {weights_path}")
    
    model = SwinIR(
        upscale=1,
        in_chans=4,           # RGB + Mask
        out_chans=3,          # RGB出力
        img_size=128,
        patch_size=1,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2.0,
        upsampler=None,
        resi_connection='1conv'
    )
    
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    logging.info("Model loaded successfully")
    return model


def process_image(model, img_path, mask_path, device):
    """単一画像の修復処理"""
    
    # 画像読み込み (BGR)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # マスク読み込み
    if mask_path and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    
    # 正規化: [0, 255] -> [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # HWC -> CHW
    img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))).float()
    mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0).float()
    
    # バッチ次元追加とデバイス移動
    img_tensor = img_tensor.unsqueeze(0).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)
    
    # 入力結合 (RGB + Mask)
    input_tensor = torch.cat([img_tensor, mask_tensor], dim=1)
    
    # 推論
    with torch.no_grad():
        output = model(input_tensor)
    
    # テンソル -> numpy: [0, 1] -> [0, 255]
    output_np = output.squeeze(0).cpu().numpy()
    output_np = np.transpose(output_np, (1, 2, 0))  # CHW -> HWC
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    
    # RGB -> BGR (OpenCV用)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    return output_bgr


def main():
    args = parse_args()
    
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
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # パス設定
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    
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
    model = load_model(args.weights, device)
    
    # モデル統計をログ
    if args.use_wandb:
        try:
            from thop import profile
            dummy_input = torch.randn(1, 4, 128, 128).to(device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            wandb.log({
                "model/FLOPs_G": flops / 1e9,
                "model/Parameters_M": params / 1e6
            })
            logging.info(f"Model FLOPs: {flops / 1e9:.2f} G, Parameters: {params / 1e6:.2f} M")
        except ImportError:
            logging.warning("thop not installed, skipping FLOPs calculation")
    
    # 画像処理
    processed_count = 0
    for img_path in tqdm(image_files, desc="Restoring images"):
        try:
            # 対応するマスクファイルを探す
            mask_path = None
            if mask_dir:
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    mask_path = mask_dir / f"{img_path.stem}.png"
                    if not mask_path.exists():
                        mask_path = None
            
            # 画像修復
            restored_img = process_image(model, img_path, mask_path, device)
            
            # 結果保存
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