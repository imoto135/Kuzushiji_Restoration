#!/usr/bin/env python3
"""
NAFNet推論スクリプト (Pattern1 TVLoss用)
学習済みモデル(TVLoss)と予測マスクを使用して損傷画像を修復します
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
        config_path: 設定ファイルのパス
        checkpoint_path: チェックポイントファイルのパス
        device: 使用するデバイス
    
    Returns:
        model: ロードされたモデル
    """
    # 設定ファイルを読み込み
    opt = parse(config_path, is_train=False)
    
    # ネットワーク設定を取得
    net_opt = opt['network_g']
    
    # モデルを構築
    model = NAFNet(
        img_channel=net_opt.get('img_channel', 4),
        width=net_opt.get('width', 32),
        middle_blk_num=net_opt.get('middle_blk_num', 12),
        enc_blk_nums=net_opt.get('enc_blk_nums', [2, 2, 4, 8]),
        dec_blk_nums=net_opt.get('dec_blk_nums', [2, 2, 2, 2]),
        out_channel=net_opt.get('out_channel', 3)
    )
    
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
    model.eval()
    model = model.to(device)
    
    print(f"✓ モデルをロードしました: {checkpoint_path}")
    print(f"  - Input channels: {net_opt.get('img_channel', 4)}")
    print(f"  - Output channels: {net_opt.get('out_channel', 3)}")
    print(f"  - Device: {device}")
    
    return model


def find_mask_path(lq_path, mask_dir):
    """
    損傷画像のファイル名から対応するマスク画像のパスを見つける
    """
    # ファイル名を取得（拡張子なし）
    lq_stem = lq_path.stem
    
    # マスクディレクトリ内で対応するファイルを探す
    for ext in ['.jpg', '.png', '.jpeg']:
        mask_path = mask_dir / f"{lq_stem}{ext}"
        if mask_path.exists():
            return mask_path
    
    return None


def process_image(model, lq_path, mask_path, device='cuda'):
    """
    1枚の画像を処理（マスク付き）
    """
    # 損傷画像を読み込み (BGR)
    lq_img = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
    if lq_img is None:
        raise ValueError(f"損傷画像の読み込みに失敗しました: {lq_path}")
    
    # マスク画像を読み込み (Grayscale)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"マスク画像の読み込みに失敗しました: {mask_path}")
    
    # サイズを合わせる
    h, w = lq_img.shape[:2]
    if mask_img.shape != (h, w):
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # BGR -> RGB
    lq_rgb = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
    
    # numpy -> tensor (0-1に正規化)
    lq_tensor = torch.from_numpy(lq_rgb).float() / 255.0
    lq_tensor = lq_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    mask_tensor = torch.from_numpy(mask_img).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
    
    # LQとマスクを結合 [3, H, W] + [1, H, W] -> [4, H, W]
    input_tensor = torch.cat([lq_tensor, mask_tensor], dim=0)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # [4, H, W] -> [1, 4, H, W]
    
    # 推論
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, list):
            output = output[-1]
    
    # tensor -> numpy (0-1 -> 0-255)
    output = output.squeeze(0).cpu().clamp(0, 1).numpy()
    output = (output * 255.0).astype(np.uint8)
    output = output.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    
    # RGB -> BGR
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output_bgr


import wandb

def main():
    parser = argparse.ArgumentParser(description='NAFNet推論スクリプト (Pattern1 TVLoss版)')
    parser.add_argument('--use_wandb', action='store_true', help='wandbにログを保存するかどうか')
    parser.add_argument('--wandb_project', type=str, default='Kuzushiji_Restoration', help='wandbプロジェクト名')
    parser.add_argument('--wandb_job_type', type=str, default='inference', help='wandbジョブタイプ')
    parser.add_argument('--log_sample_num', type=int, default=30, help='wandbにログする画像数')
    parser.add_argument(
        '--config',
        type=str,
        default='options/Kuzushiji/NAFNet-Pattern1-TVLoss.yml',
        help='設定ファイルのパス'
    )
    # ユーザー指定のパスを設定
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='experiments/NAFNet_Kuzushiji_Pattern1_TVLoss/models/net_g_295000.pth',
        help='モデルチェックポイントのパス'
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
        default='../hiragana_fulldataset_5stain/pred_mask/test',
        help='マスク画像ディレクトリ'
    )
    parser.add_argument(
        '--output-dir',
        default='../outputs/nafnet_TVLoss',
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
    print(f"NAFNet推論 (Pattern1 TVLoss: Iter 295,000)")
    print(f"{'='*60}")
    print(f"設定ファイル: {args.config}")
    print(f"モデル: {args.checkpoint}")
    print(f"入力ディレクトリ: {args.input_dir}")
    print(f"マスクディレクトリ: {args.mask_dir}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # モデルをロード
    model = load_model(args.config, args.checkpoint, args.device)
    
    # 入力画像のリストを取得
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {args.input_dir}")
    
    if not mask_dir.exists():
        raise FileNotFoundError(f"マスクディレクトリが存在しません: {args.mask_dir}")
    
    # 指定された拡張子の画像ファイルを取得
    image_files = sorted(list(input_dir.glob(f'*{args.ext}')))
    
    if len(image_files) == 0:
        print(f"Warning: 画像が見つかりませんでした (拡張子: {args.ext})")
        return
    
    print(f"\n処理する画像数: {len(image_files)}\n")

    # wandb初期化
    wandb_table = None
    log_interval = 0
    if args.use_wandb:
        try:
            wandb.init(project=args.wandb_project, job_type=args.wandb_job_type, config=vars(args))
            wandb_table = wandb.Table(columns=["Filename", "Input", "Mask", "Output"])
            if args.log_sample_num > 0:
                log_interval = max(1, len(image_files) // args.log_sample_num)
        except Exception as e:
            print(f"Wandb init failed: {e}")
            args.use_wandb = False
    
    # 統計
    processed = 0
    skipped = 0
    
    # 画像を1枚ずつ処理
    for i, lq_path in enumerate(tqdm(image_files, desc="画像を処理中")):
        try:
            # 対応するマスク画像を探す
            mask_path = find_mask_path(lq_path, mask_dir)
            
            if mask_path is None:
                print(f"\nWarning: マスクが見つかりませんでした: {lq_path.name}")
                skipped += 1
                continue
            
            # 画像を処理
            restored_img = process_image(model, lq_path, mask_path, args.device)
            
            # 出力パスを生成
            output_path = output_dir / lq_path.name
            
            # 保存
            cv2.imwrite(str(output_path), restored_img)
            
            # wandbログ記録
            if args.use_wandb and log_interval > 0 and i % log_interval == 0:
                vis_lq = cv2.imread(str(lq_path))
                vis_lq = cv2.cvtColor(vis_lq, cv2.COLOR_BGR2RGB)
                
                vis_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                h, w = vis_lq.shape[:2]
                if vis_mask.shape[:2] != (h, w):
                    vis_mask = cv2.resize(vis_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                vis_out = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                
                if wandb_table is not None:
                    wandb_table.add_data(lq_path.name, wandb.Image(vis_lq), wandb.Image(vis_mask), wandb.Image(vis_out))
                
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {lq_path.name}: {e}")
            skipped += 1
            continue

    if args.use_wandb and wandb_table is not None:
        print("Logging results to wandb...")
        try:
            wandb.log({"inference_samples": wandb_table})
            wandb.finish()
        except Exception as e:
            print(f"Wandb logging failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"✓ 完了！")
    print(f"  - 処理成功: {processed} 枚")
    print(f"  - スキップ: {skipped} 枚")
    print(f"  - 保存先: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
