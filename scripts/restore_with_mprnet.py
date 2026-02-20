#!/usr/bin/env python3
"""
MPRNetを使用したくずし字画像修復スクリプト
学習済みモデル (net_g_130000.pth) を使用して損傷画像を修復します。
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

from basicsr.models.archs.MPRNet_arch import MPRNet


def parse_args():
    parser = argparse.ArgumentParser(description="MPRNet Image Restoration for Kuzushiji")

    parser.add_argument(
        "--weights",
        type=str,
        default="models/mprnet/experiments/MPRNet_PredMask_CharbPercep/models/net_g_130000.pth",
        help="学習済みモデルの重みファイルパス"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/hiragana_fulldataset_5stain/lq/test",
        help="劣化画像のディレクトリ"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mprnet_predmask_charbpercep",
        help="修復画像の出力ディレクトリ"
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="data/hiragana_fulldataset_5stain/pred_mask/test",
        help="予測マスクのディレクトリ"
    )

    # wandb設定
    parser.add_argument("--use_wandb", action="store_true", help="wandbに結果を記録する")
    parser.add_argument("--wandb_project", type=str, default="Kuzushiji_Restoration", help="wandbプロジェクト名")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb run名")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None, help="wandbタグ")

    return parser.parse_args()


def load_model(weights_path, device):
    """MPRNetモデルをロード（mprnet_mask_charb_percep.yml の設定に合わせる）"""
    logging.info(f"Loading model from {weights_path}")

    model = MPRNet(
        in_c=4,                 # RGB + Mask
        out_c=3,
        n_feat=40,
        scale_unetfeats=20,
        scale_orsnetfeats=16,
        num_cab=4,
        kernel_size=3,
        reduction=4,
        bias=False,
    )

    checkpoint = torch.load(weights_path, map_location=device)

    # state_dict のキーを確認して取得
    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # torch.compile で学習されたモデルは _orig_mod. プレフィックスが付く
    # また total_ops / total_params などの thop 由来キーも除去する
    cleaned = {}
    for k, v in state_dict.items():
        # _orig_mod. プレフィックスを除去
        new_k = k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k
        # total_ops / total_params は重みではないのでスキップ
        if new_k.endswith("total_ops") or new_k.endswith("total_params"):
            continue
        cleaned[new_k] = v
    state_dict = cleaned

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    logging.info("Model loaded successfully")
    return model


def process_image(model, img_path, mask_path, device):
    """単一画像の修復処理"""

    # 画像読み込み (BGR -> RGB)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # マスク読み込み
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning(f"Failed to read mask: {mask_path}, using zero mask")
            mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
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

    # 入力結合 (RGB + Mask = 4ch)
    input_tensor = torch.cat([img_tensor, mask_tensor], dim=1)

    # 推論
    # MPRNet の forward は [stage3_img, stage2_img, stage1_img] を返す
    with torch.no_grad():
        outputs = model(input_tensor)

    # stage3（最終出力）を使用
    output = outputs[0]

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
            job_type="inference",
            tags=args.wandb_tags,
            config=vars(args),
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
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
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
