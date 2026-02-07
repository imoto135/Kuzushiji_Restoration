#!/usr/bin/env python3
"""
Evaluate restored images against ground-truth images.

Saves per-image PSNR/SSIM to CSV and prints summary averages.

Usage:
  python scripts/evaluate_restoration.py --gt-dir dataset_final_hiragana/gt/test \
      --pred-dir results/restormer_nomask/restored --out-csv results/restormer_nomask/metrics.csv

This script pairs files by basename (stem) and prefers .jpg over .png when both exist.
"""
import os
import sys
import argparse
import csv
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import cv2

# Ensure local `basicsr` package in the repo can be imported when running the script
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def build_map(d, allowed_exts={'.jpg', '.jpeg', '.png'}, pref_order=['.jpg', '.jpeg', '.png']):
    m = {}
    if not os.path.isdir(d):
        return m
    for fname in os.listdir(d):
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in allowed_exts:
            continue
        if stem not in m:
            m[stem] = fname
        else:
            cur_ext = os.path.splitext(m[stem])[1].lower()
            if pref_order.index(ext) < pref_order.index(cur_ext):
                m[stem] = fname
    return m


def load_img_as_float(path):
    im = Image.open(path).convert('RGB')
    arr = np.array(im).astype(np.float32) / 255.0
    return arr


def main():
    parser = argparse.ArgumentParser(description='Evaluate restored images vs GT (PSNR/SSIM)')
    parser.add_argument('--gt-dir', required=True, help='ground-truth directory (e.g. dataset/.../gt/test)')
    parser.add_argument('--pred-dir', required=True, help='predicted/restored images directory')
    parser.add_argument('--out-csv', default='', help='output CSV path (defaults to <pred-dir>/metrics.csv)')
    parser.add_argument('--resize-pred-to-gt', action='store_true', help='resize prediction to GT size before evaluation')
    parser.add_argument('--max-samples', type=int, default=0, help='limit number of samples (0 = all)')
    parser.add_argument('--mask-eval', action='store_true', help='also evaluate binary mask metrics by thresholding restored images')
    parser.add_argument('--mask-gt-dir', type=str, default='', help='ground-truth mask dir to compare against when --mask-eval is set')
    parser.add_argument('--mask-thresh', type=float, default=0.5, help='threshold in [0,1] to binarize restored images for mask evaluation')
    parser.add_argument('--mask-foreground', choices=['white', 'black'], default='white', help='Which pixel value represents the foreground in GT masks (white=255 or black=0)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    gt_map = build_map(args.gt_dir)
    pred_map = build_map(args.pred_dir)

    common = sorted(set(gt_map.keys()).intersection(set(pred_map.keys())))
    if len(common) == 0:
        logging.error('No matching filenames between GT (%s) and Pred (%s)', args.gt_dir, args.pred_dir)
        return

    if args.max_samples and args.max_samples > 0:
        common = common[:args.max_samples]

    out_csv = args.out_csv or os.path.join(args.pred_dir, 'metrics.csv')
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    results = []
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    # prepare CSV header
    header = ['stem', 'gt_file', 'pred_file', 'psnr', 'ssim']
    if args.mask_eval:
        header += ['iou', 'dice', 'precision', 'recall', 'accuracy']

    with open(out_csv, 'w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(header)
        for stem in tqdm(common, desc='Eval'):
            gt_path = os.path.join(args.gt_dir, gt_map[stem])
            pred_path = os.path.join(args.pred_dir, pred_map[stem])
            try:
                gt = load_img_as_float(gt_path)
                pred = load_img_as_float(pred_path)
                # resize pred to gt if requested
                if args.resize_pred_to_gt and pred.shape != gt.shape:
                    from PIL import Image
                    pred = np.array(Image.fromarray((pred*255).astype('uint8')).resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)).astype(np.float32)/255.0

                # calculate metrics: basicsr functions accept HWC numpy arrays (or CHW tensors)
                # ensure arrays are HWC with float values in [0,1]
                gt_hwc = gt.astype(np.float32)
                pred_hwc = pred.astype(np.float32)

                # PSNR/SSIM expect HWC numpy arrays here
                psnr = calculate_psnr(pred_hwc, gt_hwc, crop_border=0, input_order='HWC', test_y_channel=False)
                ssim = calculate_ssim(pred_hwc, gt_hwc, crop_border=0, input_order='HWC', test_y_channel=False)

                row = [stem, gt_map[stem], pred_map[stem], f'{psnr:.4f}', f'{ssim:.4f}']
                total_psnr += psnr
                total_ssim += ssim
                # optional mask evaluation
                if args.mask_eval and args.mask_gt_dir:
                    # build mask map once (lazy)
                    if 'mask_map' not in locals():
                        mask_map = build_map(args.mask_gt_dir)
                    mask_file = mask_map.get(stem, None)
                    if mask_file:
                        mask_path = os.path.join(args.mask_gt_dir, mask_file)
                        # load GT mask
                        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if m is None:
                            gt_mask = None
                        else:
                            if args.mask_foreground == 'white':
                                gt_mask = (m > 127).astype(np.uint8)
                            else:
                                # black is foreground: invert
                                gt_mask = (m < 128).astype(np.uint8)
                    else:
                        gt_mask = None

                    # create predicted mask by thresholding restored image (convert to gray)
                    try:
                        pred_vis = (pred_hwc * 255.0).astype(np.uint8)
                        pred_gray = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2GRAY)
                        thr = args.mask_thresh
                        if thr <= 1.0:
                            thr_val = int(thr * 255)
                        else:
                            thr_val = int(thr)
                        if args.mask_foreground == 'white':
                            pred_mask = (pred_gray > thr_val).astype(np.uint8)
                        else:
                            pred_mask = (pred_gray < thr_val).astype(np.uint8)
                    except Exception:
                        pred_mask = None

                    # compute metrics if GT mask exists
                    if gt_mask is not None and pred_mask is not None:
                        # resize pred mask to GT size if needed
                        if pred_mask.shape != gt_mask.shape:
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
                    else:
                        iou = dice = precision = recall = accuracy = ''

                    row += [f'{iou:.4f}' if isinstance(iou, float) else iou,
                            f'{dice:.4f}' if isinstance(dice, float) else dice,
                            f'{precision:.4f}' if isinstance(precision, float) else precision,
                            f'{recall:.4f}' if isinstance(recall, float) else recall,
                            f'{accuracy:.4f}' if isinstance(accuracy, float) else accuracy]

                writer.writerow(row)
                count += 1
            except Exception as e:
                logging.exception('Failed to evaluate %s: %s', stem, e)

    if count > 0:
        logging.info('Evaluated %d images. Mean PSNR: %.4f, Mean SSIM: %.4f', count, total_psnr/count, total_ssim/count)
    else:
        logging.warning('No images evaluated')

    logging.info('Per-image metrics saved to %s', out_csv)


if __name__ == '__main__':
    main()
