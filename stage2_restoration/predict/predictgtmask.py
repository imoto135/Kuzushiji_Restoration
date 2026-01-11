#!/usr/bin/env python3
"""
LQ + マスク から Restormer で修復して、その出力を評価するラッパー。
内部で predictgtmask.py を2回呼び出します:
  1) 推論 (weights を指定して LQ+mask -> restored を作成)
  2) 評価 (GT, restored, mask を使って Masked PSNR/SSIM/LPIPS を計算)

使い方例:
python stage2_restoration/predict/run_restore_and_eval.py \
  --weights /path/to/model.pth \
  --data-dir datasets/hiragana_dataset \
  --lq-dir lq/test \
  --mask-dir gt_mask/test \
  --gt-dir gt/test \
  --out-dir outputs/restored_test \
  --image-size 128 \
  --batch-size 8 \
  --log-wandb --wandb-project Kuzushiji_Restoration
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PREDICT_SCRIPT = os.path.join(ROOT, 'stage2_restoration', 'predict', 'predictgtmask.py')

def call(cmd_list):
    print("Running:", " ".join(cmd_list))
    p = subprocess.run(cmd_list)
    if p.returncode != 0:
        raise SystemExit(f"Command failed (exit {p.returncode}): {' '.join(cmd_list)}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--data-dir', default='datasets/hiragana_dataset')
    p.add_argument('--lq-dir', default='lq/test')
    p.add_argument('--mask-dir', default='gt_mask/test')
    p.add_argument('--gt-dir', default='gt/test')
    p.add_argument('--out-dir', default='outputs/restored_test')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--use-amp', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--log-wandb', action='store_true', help='W&B にログ')
    p.add_argument('--no-wandb', action='store_true', help='W&B を無効化（評価時）')
    p.add_argument('--wandb-project', default='Restormer')
    p.add_argument('--wandb-entity', default=None)
    args = p.parse_args()

    # 1) 推論ステップ: predictgtmask.py を呼ぶ（weights -> out_dir）
    infer_cmd = [
        sys.executable, PREDICT_SCRIPT,
        '--weights', args.weights,
        '--data-dir', args.data_dir,
        '--lq-dir', args.lq_dir,
        '--mask-dir', args.mask_dir,
        '--out-dir', args.out_dir,
        '--image-size', str(args.image_size),
        '--batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers)
    ]
    if args.use_amp:
        infer_cmd.append('--use-amp')
    if args.cpu:
        infer_cmd.append('--cpu')
    # 予測段階で wandb ログしたければこのフラグを付ける
    if args.log_wandb:
        infer_cmd.append('--log-wandb')
        if args.wandb_project:
            infer_cmd += ['--wandb-project', args.wandb_project]
        if args.wandb_entity:
            infer_cmd += ['--wandb-entity', args.wandb_entity]

    os.makedirs(args.out_dir, exist_ok=True)
    call(infer_cmd)

    # 2) 評価ステップ: 同じスクリプトを評価モードで呼ぶ（GT, restored, mask）
    eval_cmd = [
        sys.executable, PREDICT_SCRIPT,
        '--gt-dir', os.path.join(args.data_dir, args.gt_dir) if not os.path.isabs(args.gt_dir) else args.gt_dir,
        '--restored-dir', args.out_dir,
        '--mask-dir', os.path.join(args.data_dir, args.mask_dir) if not os.path.isabs(args.mask_dir) else args.mask_dir,
        '--use-wandb' if args.log_wandb and not args.no_wandb else '--no-wandb'
    ]

    # wandb 設定（評価ログ）
    if args.log_wandb and not args.no_wandb:
        eval_cmd += ['--wandb-project', args.wandb_project]
        if args.wandb_entity:
            eval_cmd += ['--wandb-entity', args.wandb_entity]

    call(eval_cmd)
    print("Restore + Evaluate done. Restored outputs in:", args.out_dir)

if __name__ == '__main__':
    main()