#!/usr/bin/env python3
"""
NAFNet推論スクリプト (Mask版) - 修正版
学習済みモデル(.pth)と予測マスクを使用して損傷画像を修復します
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import wandb

# basicsr モジュールをインポート
sys.path.insert(0, os.path.dirname(__file__))
from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.utils.options import parse


def load_model(config_path, checkpoint_path, device='cuda'):
    """
    NAFNetモデルをロード
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {checkpoint_path}")
    
    # 拡張子のチェック
    if not str(checkpoint_path).endswith('.pth'):
        raise ValueError(f"エラー: --checkpoint には .pth ファイルを指定してください。現在の入力: {checkpoint_path}")

    # 設定ファイルを読み込み
    opt = parse(config_path, is_train=False)
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
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"致命的なエラー: モデルの読み込みに失敗しました。ファイルが破損しているか、テキストファイル(YAML等)を指定している可能性があります。")
        raise e
    
    # state_dictを取得
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # module. および _orig_mod. プレフィックスの削除
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'total_ops' in k or 'total_params' in k:
            continue
            
        # torch.compile や DataParallel によるプレフィックスを除去
        name = k
        if name.startswith('module.'):
            name = name[7:]
        if name.startswith('_orig_mod.'):
            name = name[10:]
            
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model = model.to(device)
    
    print(f"✓ モデルをロードしました。")
    return model


def find_mask_path(lq_path, mask_dir):
    """損傷画像に対応するマスク画像を探す"""
    lq_stem = lq_path.stem
    for ext in ['.jpg', '.png', '.jpeg']:
        mask_path = mask_dir / f"{lq_stem}{ext}"
        if mask_path.exists():
            return mask_path
    return None


def process_image(model, lq_path, mask_path, device='cuda'):
    """1枚の画像を処理（マスク付き）"""
    lq_img = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if lq_img is None or mask_img is None:
        return None

    h, w = lq_img.shape[:2]
    if mask_img.shape != (h, w):
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    lq_rgb = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
    lq_tensor = torch.from_numpy(lq_rgb).float() / 255.0
    lq_tensor = lq_tensor.permute(2, 0, 1)
    
    mask_tensor = torch.from_numpy(mask_img).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0)
    
    input_tensor = torch.cat([lq_tensor, mask_tensor], dim=0)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, list):
            output = output[-1]
    
    output = output.squeeze(0).cpu().clamp(0, 1).numpy()
    output = (output * 255.0).astype(np.uint8)
    output = output.transpose(1, 2, 0)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output_bgr


def main():
    parser = argparse.ArgumentParser(description='NAFNet推論スクリプト (Mask版)')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='Kuzushiji_Restoration')
    parser.add_argument('--log_sample_num', type=int, default=30)
    
    # 必須引数に変更
    parser.add_argument('--config', type=str, required=True, help='設定ファイル(.yml)のパス')
    parser.add_argument('--checkpoint', type=str, required=True, help='モデル重み(.pth)のパス')
    
    parser.add_argument('--input-dir', type=str, default='../hiragana_fulldataset_5stain/lq/test')
    parser.add_argument('--mask-dir', type=str, default='../hiragana_fulldataset_5stain/pred_mask/test')
    parser.add_argument('--output-dir', type=str, default='../outputs/nafnet_inference_result')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--ext', type=str, default='.jpg')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルをロード
    model = load_model(args.config, args.checkpoint, args.device)
    
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    image_files = sorted(list(input_dir.glob(f'*{args.ext}')))
    
    if not image_files:
        print(f"画像が見つかりません: {args.input_dir}")
        return

    if args.use_wandb:
        wandb.init(project=args.wandb_project, job_type='inference', config=vars(args))
        wandb_table = wandb.Table(columns=["Filename", "Input", "Mask", "Output"])
        log_interval = max(1, len(image_files) // args.log_sample_num)

    processed = 0
    for i, lq_path in enumerate(tqdm(image_files, desc="Processing")):
        mask_path = find_mask_path(lq_path, mask_dir)
        if mask_path is None:
            continue
            
        restored_img = process_image(model, lq_path, mask_path, args.device)
        if restored_img is None:
            continue
            
        output_path = output_dir / lq_path.name
        cv2.imwrite(str(output_path), restored_img)
        
        if args.use_wandb and i % log_interval == 0:
            vis_lq = cv2.cvtColor(cv2.imread(str(lq_path)), cv2.COLOR_BGR2RGB)
            vis_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            vis_out = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            wandb_table.add_data(lq_path.name, wandb.Image(vis_lq), wandb.Image(vis_mask), wandb.Image(vis_out))
            
        processed += 1

    if args.use_wandb:
        wandb.log({"inference_samples": wandb_table})
        wandb.finish()
    
    print(f"\n✓ 完了: {processed} 枚の画像を保存しました -> {args.output_dir}")


if __name__ == '__main__':
    main()