#!/usr/bin/env python3
"""
NAFNetを使用したくずし字画像修復スクリプト (マスクなし)
最高精度を記録した学習済みモデル (net_g_160000.pth) を使用して損傷画像を修復します。
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

from basicsr.models.archs.NAFNet_arch import NAFNet


def parse_args():
    parser = argparse.ArgumentParser(description="NAFNet Image Restoration (No Mask) for Kuzushiji")
    
    parser.add_argument(
        "--weights",
        type=str,
        default="models/nafnet/experiments/NAFNet_Kuzushiji_NoMask_CharbPercep/models/net_g_160000.pth",
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
        default="outputs/nafnet_nomask_charbpercep",
        help="Directory to save restored images"
    )
    
    # wandb設定
    parser.add_argument('--use_wandb', action='store_true', help='wandbに結果を記録する')
    parser.add_argument('--wandb_project', type=str, default='Kuzushiji_Restoration', help='wandbプロジェクト名')
    parser.add_argument('--wandb_name', type=str, default='NAFNet_NoMask_Inference', help='wandb run名')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=['inference', 'nafnet'], help='wandbタグ')
    
    return parser.parse_args()


def load_model(weights_path, device):
    """NAFNetモデルをロード"""
    logging.info(f"Loading model from {weights_path}")
    
    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 余分なキーの削除と 'module.' の削除
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'total_ops' in k or 'total_params' in k:
            continue
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model = model.to(device)
    
    logging.info("Model loaded successfully")
    return model


def process_image(model, img_path, device):
    """単一画像の修復処理 (マスクなし)"""
    
    # 画像読み込み (BGR)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 正規化: [0, 255] -> [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # HWC -> CHW
    img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))).float()
    
    # バッチ次元追加とデバイス移動
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 推論 (No Mask)
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, list):
            output = output[-1]
    
    # テンソル -> numpy: [0, 1] -> [0, 255]
    output_np = output.squeeze(0).cpu().clamp(0, 1).numpy()
    output_np = np.transpose(output_np, (1, 2, 0))  # CHW -> HWC
    output_np = (output_np * 255.0).astype(np.uint8)
    
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
    model = load_model(args.weights, device)
    
    # モデル統計をログ
    if args.use_wandb:
        try:
            from thop import profile
            dummy_input = torch.randn(1, 3, 128, 128).to(device)
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
            # 画像修復 (マスクなし)
            restored_img = process_image(model, img_path, device)
            
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
