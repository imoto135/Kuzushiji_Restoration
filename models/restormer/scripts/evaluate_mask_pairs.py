#!/usr/bin/env python3
"""Compare GT masks and predicted masks one-to-one and save metrics + side-by-side images.

Usage:
  python scripts/evaluate_mask_pairs.py --gt-mask dataset_final_hiragana/mask_gt/test \
      --pred-mask dataset_final_hiragana/mask_random_prediction/test --out results/mask_comparison_mask_gt_vs_mask_random_prediction

Outputs:
 - <out>/per_image.csv
 - <out>/summary.csv
 - side-by-side images in <out>/images/

Metrics: IoU, Dice, Precision, Recall, Accuracy (same definitions as evaluate_samples_vs_gt.py)
"""
import os
import argparse
from glob import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_file_with_exts(dirpath, stem, exts=('.png', '.jpg', '.jpeg')):
    if not os.path.isdir(dirpath):
        return None
    # prefer exact match first
    for e in exts:
        p = os.path.join(dirpath, stem + e)
        if os.path.isfile(p):
            return p
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


def mask_to_vis(m):
    if m is None:
        return None
    mm = (m * 255).astype(np.uint8) if m.max() <= 1 else m.astype(np.uint8)
    vis = cv2.cvtColor(mm, cv2.COLOR_GRAY2BGR)
    # colorize: GT in green overlay, pred in red overlay? For side-by-side we just show grayscale
    return vis


def make_side_by_side(gt_vis, pred_vis, meta_text=None):
    # ensure same size
    if gt_vis is None and pred_vis is None:
        return None
    if gt_vis is None:
        h, w = pred_vis.shape[:2]
        gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    if pred_vis is None:
        h, w = gt_vis.shape[:2]
        pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
    if gt_vis.shape[:2] != pred_vis.shape[:2]:
        pred_vis = cv2.resize(pred_vis, (gt_vis.shape[1], gt_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    comp = np.hstack([gt_vis, pred_vis])
    if meta_text:
        overlay = comp.copy()
        cv2.rectangle(overlay, (0,0), (comp.shape[1], 26), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, comp, 0.6, 0, comp)
        cv2.putText(comp, meta_text, (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return comp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt-mask', required=True, help='GT mask dir')
    p.add_argument('--pred-mask', required=True, help='Predicted mask dir')
    p.add_argument('--out', default='results/gt vs deeplabv3p_noembed', help='Output dir')
    p.add_argument('--binary-thresh', type=int, default=127)
    p.add_argument('--mask-foreground', choices=['white', 'black'], default='white',
                   help='Which pixel value represents the foreground (文字). default=white')
    args = p.parse_args()

    gt_dir = args.gt_mask
    pred_dir = args.pred_mask
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    img_out = os.path.join(out_dir, 'images')
    os.makedirs(img_out, exist_ok=True)

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    rows = []
    for fn in tqdm(gt_files, desc='Comparing masks'):
        stem, _ = os.path.splitext(fn)
        gt_path = find_file_with_exts(gt_dir, stem)
        pred_path = find_file_with_exts(pred_dir, stem)
        if gt_path is None:
            # skip
            continue
        gt_mask = load_mask_bin(gt_path, thresh=args.binary_thresh)
        pred_mask = load_mask_bin(pred_path, thresh=args.binary_thresh) if pred_path else None
        # if masks use black as foreground (文字 = 0), invert to make foreground==1
        if args.mask_foreground == 'black':
            if gt_mask is not None:
                gt_mask = (1 - gt_mask).astype(np.uint8)
            if pred_mask is not None:
                pred_mask = (1 - pred_mask).astype(np.uint8)
        if gt_mask is None:
            continue
        if pred_mask is None:
            # treat as empty
            pred_mask = np.zeros_like(gt_mask)

        iou, dice, precision, recall, accuracy = compute_mask_metrics(gt_mask, pred_mask)
        rows.append([fn, os.path.basename(pred_path) if pred_path else '', iou, dice, precision, recall, accuracy])

        # create side-by-side visualization
        gt_vis = mask_to_vis(gt_mask)
        pred_vis = mask_to_vis(pred_mask)
        meta = f"IoU={iou:.3f} D={dice:.3f} P={precision:.3f} R={recall:.3f}"
        comp = make_side_by_side(gt_vis, pred_vis, meta_text=meta)
        outname = os.path.join(img_out, stem + '.png')
        if comp is not None:
            cv2.imwrite(outname, comp)

    df = pd.DataFrame(rows, columns=['gt_filename', 'pred_filename', 'iou', 'dice', 'precision', 'recall', 'accuracy'])
    # save per-image CSV
    per_csv = os.path.join(out_dir, 'per_image.csv')
    df.to_csv(per_csv, index=False)

    # summary
    summary = {}
    total = df.shape[0]
    summary['total'] = int(total)
    for c in ['iou', 'dice', 'precision', 'recall', 'accuracy']:
        valid = pd.to_numeric(df[c], errors='coerce').dropna()
        summary[f'{c}_count'] = int(valid.shape[0])
        summary[f'{c}_mean'] = float(round(valid.mean(), 3)) if valid.shape[0] > 0 else None
        summary[f'{c}_std'] = float(round(valid.std(ddof=0), 3)) if valid.shape[0] > 0 else None

    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(out_dir, 'summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    print('Saved per-image CSV to', per_csv)
    print('Saved summary to', summary_csv)
    print('Saved comparison images to', img_out)

if __name__ == '__main__':
    main()
