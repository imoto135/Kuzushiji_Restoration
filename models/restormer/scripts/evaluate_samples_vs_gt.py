#!/usr/bin/env python3
"""
Evaluate restored images in `results/samples` against GT in `dataset_final_hiragana/gt/test` one-to-one.
Saves per-image CSV (`results/samples/evaluation.csv`) and summary (`results/samples/evaluation_summary.csv`).

Metrics:
 - PSNR, SSIM (full-image)
 - IoU, Dice, Precision, Recall, Accuracy (mask-based) if --gt-mask and --pred-mask provided

Usage:
  python scripts/evaluate_samples_vs_gt.py

This script expects that restored outputs were saved with the same basenames as GT (or at least contain the GT stem). It will iterate over GT files and look for a matching file in the restored dir.
"""
import os
import argparse
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def find_file_with_exts(dirpath, stem, exts=('.png', '.jpg', '.jpeg')):
    if not os.path.isdir(dirpath):
        return None
    # prefer exact match first
    for e in exts:
        p = os.path.join(dirpath, stem + e)
        if os.path.isfile(p):
            return p
    # fallback: any file whose stem equals or contains the stem
    for f in os.listdir(dirpath):
        name, _ = os.path.splitext(f)
        if name == stem or name.startswith(stem) or stem.startswith(name):
            return os.path.join(dirpath, f)
    return None


def load_mask_bin(path, thresh=127):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.max() > 1:
        m = (m > thresh).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)
    return m


def compute_mask_metrics(gt_mask, pred_mask):
    if gt_mask.shape != pred_mask.shape:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_mask = (pred_mask > 0).astype(np.uint8)

    tp = int(np.logical_and(gt_mask == 1, pred_mask == 1).sum())
    tn = int(np.logical_and(gt_mask == 0, pred_mask == 0).sum())
    fp = int(np.logical_and(gt_mask == 0, pred_mask == 1).sum())
    fn = int(np.logical_and(gt_mask == 1, pred_mask == 0).sum())

    union = tp + fp + fn
    iou = (tp / union) if union > 0 else 0.0

    denom = (2 * tp + fp + fn)
    dice = (2 * tp / denom) if denom > 0 else 0.0

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    total = tp + tn + fp + fn
    accuracy = ((tp + tn) / total) if total > 0 else 0.0

    return iou, dice, precision, recall, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='dataset_final_hiragana/gt/test', help='GT images directory (iterate over these)')
    parser.add_argument('--restored', default='results/deeplabv3p_b7/pred_masks', help='Restored images directory (one-to-one with GT)')
    parser.add_argument('--gt-mask', default='dataset_final_hiragana/gt_mask/test', help='GT mask dir for mask metrics (optional)')
    parser.add_argument('--pred-mask', default=None, help='Predicted mask dir for mask metrics (optional)')
    parser.add_argument('--out', default='results/deeplabv3p_b7/evaluation.csv', help='Output CSV path (defaults to <restored>/evaluation.csv)')
    parser.add_argument('--binary-thresh', type=int, default=127, help='Threshold for mask binarization')
    args = parser.parse_args()

    gt_dir = args.gt
    restored_dir = args.restored
    gt_mask_dir = args.gt_mask
    pred_mask_dir = args.pred_mask

    if args.out is None:
        out_csv = os.path.join(restored_dir, 'evaluation.csv')
    else:
        out_csv = args.out

    if not os.path.isdir(gt_dir):
        raise SystemExit(f'GT dir not found: {gt_dir}')
    if not os.path.isdir(restored_dir):
        raise SystemExit(f'Restored dir not found: {restored_dir}')

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not gt_files:
        raise SystemExit(f'No GT images found in {gt_dir}')

    rows = []
    missing = 0
    for fn in tqdm(gt_files, desc='Evaluating'):
        stem, _ = os.path.splitext(fn)
        gt_path = os.path.join(gt_dir, fn)
        # find matching restored by stem in restored_dir
        restored_path = find_file_with_exts(restored_dir, stem)
        if restored_path is None:
            # record missing and skip
            missing += 1
            print(f'Warning: restored image not found for GT {fn}, skipping')
            continue
        gt = cv2.imread(gt_path)
        restored = cv2.imread(restored_path)
        if gt is None or restored is None:
            print(f'Warning: cannot read GT or restored for {fn}, skipping')
            continue

        # if restored is composite, try to extract rightmost tile matching GT width
        r_h, r_w = restored.shape[0], restored.shape[1]
        g_h, g_w = gt.shape[0], gt.shape[1]
        restored_for_eval = restored
        if r_w >= g_w and g_w > 0 and r_w % g_w == 0 and (r_w // g_w) >= 2:
            parts = r_w // g_w
            restored_for_eval = restored[:, (parts - 1) * g_w: parts * g_w]
        else:
            if r_w % r_h == 0 and (r_w // r_h) in (2, 3):
                parts = r_w // r_h
                restored_for_eval = restored[:, (parts - 1) * r_h: parts * r_h]

        if restored_for_eval.shape[0] != g_h or restored_for_eval.shape[1] != g_w:
            try:
                restored_for_eval = cv2.resize(restored_for_eval, (g_w, g_h), interpolation=cv2.INTER_AREA)
            except Exception:
                pass

        try:
            psnr_v = cv2.PSNR(gt, restored_for_eval)
        except Exception:
            psnr_v = None
        try:
            ssim_v = ssim(gt, restored_for_eval, channel_axis=-1, data_range=255)
        except Exception:
            ssim_v = None

        iou = dice = precision = recall = accuracy = None
        if gt_mask_dir and pred_mask_dir:
            gt_mask_path = find_file_with_exts(gt_mask_dir, stem)
            pred_mask_path = find_file_with_exts(pred_mask_dir, stem)
            if gt_mask_path and pred_mask_path:
                gt_mask = load_mask_bin(gt_mask_path, thresh=args.binary_thresh)
                pred_mask = load_mask_bin(pred_mask_path, thresh=args.binary_thresh)
                if gt_mask is not None and pred_mask is not None:
                    iou, dice, precision, recall, accuracy = compute_mask_metrics(gt_mask, pred_mask)

        rows.append([fn, os.path.basename(restored_path), psnr_v, ssim_v, iou, dice, precision, recall, accuracy])

    df = pd.DataFrame(rows, columns=['gt_filename', 'restored_filename', 'psnr', 'ssim', 'iou', 'dice', 'precision', 'recall', 'accuracy'])
    num_cols = ['psnr', 'ssim', 'iou', 'dice', 'precision', 'recall', 'accuracy']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_round = df.copy()
    for c in num_cols:
        if c in df_round.columns:
            df_round[c] = df_round[c].round(3)
    df_round.to_csv(out_csv, index=False)
    print('Saved evaluation to', out_csv)

    # summary
    summary = {}
    total = len(gt_files)
    summary['total_gt_images'] = int(total)
    summary['matched'] = int(df.shape[0])
    summary['missing'] = int(total - df.shape[0])
    for c in ['psnr', 'ssim']:
        if c in df.columns:
            valid = df[c].dropna()
            summary[f'{c}_count'] = int(valid.shape[0])
            summary[f'{c}_mean'] = float(round(valid.mean(), 3)) if valid.shape[0] > 0 else None
            summary[f'{c}_std'] = float(round(valid.std(ddof=0), 3)) if valid.shape[0] > 0 else None
    for c in ['iou', 'dice', 'precision', 'recall', 'accuracy']:
        if c in df.columns:
            valid = df[c].dropna()
            summary[f'{c}_count'] = int(valid.shape[0])
            summary[f'{c}_mean'] = float(round(valid.mean(), 3)) if valid.shape[0] > 0 else None
            summary[f'{c}_std'] = float(round(valid.std(ddof=0), 3)) if valid.shape[0] > 0 else None

    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(os.path.dirname(out_csv), 'evaluation_summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    # print compact
    print('\nEvaluation summary:')
    print(f"  total GT images: {summary['total_gt_images']}")
    print(f"  matched restored: {summary['matched']}")
    print(f"  missing restored: {summary['missing']}")
    def print_stat(c):
        if f'{c}_count' in summary:
            print(f"  {c}: count={summary[f'{c}_count']}, mean={summary.get(f'{c}_mean')}, std={summary.get(f'{c}_std')}")
    print_stat('psnr')
    print_stat('ssim')
    for m in ['iou', 'dice', 'precision', 'recall', 'accuracy']:
        print_stat(m)
    print('\nSaved summary to', summary_csv)


if __name__ == '__main__':
    main()
