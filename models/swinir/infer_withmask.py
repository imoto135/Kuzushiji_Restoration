#!/usr/bin/env python3
"""
SwinIR推論スクリプト (Mask版)
マスク情報を使用して損傷画像を修復します
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm

# basicsr モジュールをインポート
sys.path.insert(0, os.path.dirname(__file__))
from basicsr.models.archs.SwinIR_arch import SwinIR


def load_model(checkpoint_path, img_size=128, in_chans=4, device='cuda'):
    """
    SwinIRモデルをロード（マスク入力対応）
    
    Args:
        checkpoint_path: チェックポイントファイルのパス
        img_size: 入力画像サイズ
        in_chans: 入力チャネル数（RGB + Mask = 4）
        device: 使用するデバイス ('cuda' or 'cpu')
    
    Returns:
        model: ロードされたモデル
    """
    # モデルを構築（入力チャネル数を4に設定）
    model = SwinIR(
        img_size=img_size,
        patch_size=1,
        in_chans=in_chans,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.,
        upscale=1,
        img_range=1.,
        upsampler='',
        resi_connection='1conv'
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # チェックポイントをロード
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # state_dictを取得
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # キー名の調整
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'total_ops' in k or 'total_params' in k:
                continue
                
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print(f"✓ モデルをロードしました: {checkpoint_path}")
    else:
        print(f"⚠ チェックポイントが見つかりません。未学習モデルを使用します。")
    
    model.eval()
    model = model.to(device)
    
    print(f"  - Input channels: {in_chans}")
    print(f"  - Device: {device}")
    
    return model


def process_image(model, img_path, mask_path, device='cuda'):
    """
    1枚の画像を処理（マスク付き）
    
    Args:
        model: SwinIRモデル
        img_path: 入力画像のパス
        mask_path: マスク画像のパス
        device: 使用するデバイス
    
    Returns:
        restored_img: 修復された画像 (numpy array, BGR, uint8)
    """
    # 画像を読み込み (BGR)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"画像の読み込みに失敗しました: {img_path}")
    
    # マスクを読み込み (グレースケール)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"マスクの読み込みに失敗しました: {mask_path}")
    
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # numpy -> tensor (0-1に正規化)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    mask_tensor = torch.from_numpy(mask).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
    
    # RGB + Mask を結合
    input_tensor = torch.cat([img_tensor, mask_tensor], dim=0).unsqueeze(0)  # [1, 4, H, W]
    input_tensor = input_tensor.to(device)
    
    # 推論
    with torch.no_grad():
        output = model(input_tensor)
        
        # リストで返される場合は最後の要素を使用
        if isinstance(output, list):
            output = output[-1]
    
    # tensor -> numpy (0-1 -> 0-255)
    output = output.squeeze(0).cpu().clamp(0, 1).numpy()
    output = (output * 255.0).astype(np.uint8)
    output = output.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    
    # RGB -> BGR
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output_bgr


def main():
    parser = argparse.ArgumentParser(description='SwinIR推論スクリプト (Mask版)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='モデルチェックポイントのパス（オプション）'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../hiragana_fulldataset_5stain/lq/test',
        help='入力画像ディレクトリ'
    )
    parser.add_argument(
        '--mask-dir',
        type=str,
        default='../hiragana_fulldataset_5stain/gt_mask/test',
        help='マスク画像ディレクトリ'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../outputs/swinir_withmask',
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='使用するデバイス'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=128,
        help='入力画像サイズ'
    )
    
    args = parser.parse_args()
    
    # デバイスの設定
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDAが利用できません。CPUを使用します。")
        args.device = 'cpu'
    
    # モデルをロード
    print("\n=== モデルのロード ===")
    model = load_model(args.checkpoint, img_size=args.img_size, in_chans=4, device=args.device)
    
    # 入力ディレクトリの確認
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    if not input_dir.exists():
        raise ValueError(f"入力ディレクトリが存在しません: {input_dir}")
    if not mask_dir.exists():
        raise ValueError(f"マスクディレクトリが存在しません: {mask_dir}")
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルのリストを取得
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f'*{ext}')))
        image_files.extend(list(input_dir.glob(f'*{ext.upper()}')))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"⚠ 入力ディレクトリに画像ファイルが見つかりません: {input_dir}")
        return
    
    print(f"\n=== 推論の実行 ===")
    print(f"入力ディレクトリ: {input_dir}")
    print(f"マスクディレクトリ: {mask_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"画像数: {len(image_files)}")
    
    # 各画像を処理
    for img_path in tqdm(image_files, desc="処理中"):
        try:
            # 対応するマスクファイルを探す
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                print(f"\n⚠ マスクが見つかりません: {mask_path}")
                continue
            
            # 画像を処理
            restored_img = process_image(model, img_path, mask_path, device=args.device)
            
            # 出力パスを生成
            output_path = output_dir / img_path.name
            
            # 画像を保存
            cv2.imwrite(str(output_path), restored_img)
            
        except Exception as e:
            print(f"\n✗ エラー: {img_path.name} - {str(e)}")
            continue
    
    print(f"\n✓ 完了しました！")
    print(f"出力先: {output_dir}")


if __name__ == '__main__':
    main()
