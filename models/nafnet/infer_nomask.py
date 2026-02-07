#!/usr/bin/env python3
"""
NAFNet推論スクリプト (No Mask版)
学習済みモデルを使用して損傷画像を修復します
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
from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.utils.options import parse


def load_model(config_path, checkpoint_path, device='cuda'):
    """
    NAFNetモデルをロード
    
    Args:
        config_path: 設定ファイルのパス (例: options/Kuzushiji/nomask.yml)
        checkpoint_path: チェックポイントファイルのパス
        device: 使用するデバイス ('cuda' or 'cpu')
    
    Returns:
        model: ロードされたモデル
    """
    # 設定ファイルを読み込み
    opt = parse(config_path, is_train=False)
    
    # ネットワーク設定を取得
    net_opt = opt['network_g']
    
    # モデルを構築
    model = NAFNet(
        img_channel=net_opt.get('img_channel', 3),
        width=net_opt.get('width', 32),
        middle_blk_num=net_opt.get('middle_blk_num', 12),
        enc_blk_nums=net_opt.get('enc_blk_nums', [2, 2, 4, 8]),
        dec_blk_nums=net_opt.get('dec_blk_nums', [2, 2, 2, 2]),
        out_channel=net_opt.get('out_channel', 3)
    )
    
    # チェックポイントをロード
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # state_dictを取得（キーに 'params' が含まれている場合と含まれていない場合に対応）
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # キー名の調整（'module.' プレフィックスを削除、および不要なキーを除外）
    new_state_dict = {}
    for k, v in state_dict.items():
        # 余分なキー（total_ops, total_paramsなど）を除外
        if 'total_ops' in k or 'total_params' in k:
            continue
            
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model = model.to(device)
    
    print(f"✓ モデルをロードしました: {checkpoint_path}")
    print(f"  - Input channels: {net_opt.get('img_channel', 3)}")
    print(f"  - Output channels: {net_opt.get('out_channel', 3)}")
    print(f"  - Device: {device}")
    
    return model


def process_image(model, img_path, device='cuda'):
    """
    1枚の画像を処理
    
    Args:
        model: NAFNetモデル
        img_path: 入力画像のパス
        device: 使用するデバイス
    
    Returns:
        restored_img: 修復された画像 (numpy array, BGR, uint8)
    """
    # 画像を読み込み (BGR)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"画像の読み込みに失敗しました: {img_path}")
    
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # numpy -> tensor (0-1に正規化)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
    img_tensor = img_tensor.to(device)
    
    # 推論
    with torch.no_grad():
        output = model(img_tensor)
        
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
    parser = argparse.ArgumentParser(description='NAFNet推論スクリプト (No Mask)')
    parser.add_argument(
        '--config',
        type=str,
        default='options/Kuzushiji/nomask.yml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='experiments/NAFNet_Kuzushiji_NoMask/models/net_g_110000.pth',
        help='モデルチェックポイントのパス'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../hiragana_fulldataset_5stain/lq/test',
        help='入力画像ディレクトリ'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../outputs/nafnet_nomask',
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
        '--ext',
        type=str,
        default='.jpg',
        help='入力画像の拡張子（例: .jpg, .png）'
    )
    
    args = parser.parse_args()
    
    # デバイスの確認
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDAが利用できません。CPUを使用します。")
        args.device = 'cpu'
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"NAFNet推論 (No Mask版)")
    print(f"{'='*60}")
    print(f"設定ファイル: {args.config}")
    print(f"モデル: {args.checkpoint}")
    print(f"入力ディレクトリ: {args.input_dir}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # モデルをロード
    model = load_model(args.config, args.checkpoint, args.device)
    
    # 入力画像のリストを取得
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {args.input_dir}")
    
    # 指定された拡張子の画像ファイルを取得
    image_files = sorted(list(input_dir.glob(f'*{args.ext}')))
    
    if len(image_files) == 0:
        print(f"Warning: 画像が見つかりませんでした (拡張子: {args.ext})")
        return
    
    print(f"\n処理する画像数: {len(image_files)}\n")
    
    # 画像を1枚ずつ処理
    for img_path in tqdm(image_files, desc="画像を処理中"):
        try:
            # 画像を処理
            restored_img = process_image(model, img_path, args.device)
            
            # 出力パスを生成
            output_path = output_dir / img_path.name
            
            # 保存
            cv2.imwrite(str(output_path), restored_img)
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ 完了！修復画像を保存しました: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
