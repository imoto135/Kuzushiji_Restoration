#!/usr/bin/env python3
"""Visualize low-IoU failure cases.
Reads an evaluation CSV (created by scripts/evaluate_samples_vs_gt.py), selects the N rows with lowest IoU,
and creates 2x2 composite images (GT | restored \n GT-mask | pred-mask) with IoU/Precision/Recall overlayed.

Usage:
  python scripts/visualize_failures.py --eval results/samples/evaluation.csv --gt dataset_final_hiragana/gt/test \
      --restored results/samples --gt-mask dataset_final_hiragana/mask_random_gt/test \
      --pred-mask results/samples_pred_masks_binarized_t110_m30 --out results/failures_low_iou --n 50

If a mask file is missing for a sample, it will still save the composite with blanks.
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_file_with_exts(dirpath, stem, exts=('.png', '.jpg', '.jpeg')):
    if dirpath is None or not os.path.isdir(dirpath):
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


def load_img(path, target_size=None):
    if path is None or not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def mask_to_vis(m):
    # m: single-channel 0/1 or 0..255 -> convert to 3-channel RGB for visualization
    if m is None:
        return None
    import numpy as np
    if m.dtype != np.uint8:
        m = (m.astype(np.uint8))
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = m.astype(np.uint8)
    vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    # make mask appear red-ish: color map where mask=255 -> white, leave as is for now
    return vis


def make_composite(gt, restored, gt_mask, pred_mask, meta_text=None):
    # All inputs are BGR images or None. We'll resize everything to GT size if available, else restored.
    base = None
    if gt is not None:
        h, w = gt.shape[:2]
        base = (w, h)
    elif restored is not None:
        h, w = restored.shape[:2]
        base = (w, h)
    else:
        # fallback
        base = (256, 256)
        w, h = base

    def prepare(img):
        if img is None:
            return np.zeros((h, w, 3), dtype=np.uint8) + 255
        if img.shape[0] != h or img.shape[1] != w:
            try:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            except Exception:
                img = cv2.resize(img, (w, h))
        # ensure 3 channels
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    gt_p = prepare(gt)
    restored_p = prepare(restored)
    gt_mask_p = prepare(gt_mask)
    pred_mask_p = prepare(pred_mask)

    pad = 6
    top = np.hstack([gt_p, restored_p])
    bottom = np.hstack([gt_mask_p, pred_mask_p])
    composite = np.vstack([top, bottom])

    # overlay meta text
    if meta_text:
        # draw a translucent rectangle
        overlay = composite.copy()
        cv2.rectangle(overlay, (0,0), (composite.shape[1], 28), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, composite, 0.6, 0, composite)
        x = 6
        y = 20
        cv2.putText(composite, meta_text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # draw small labels on each quadrant
    h2 = composite.shape[0] // 2
    w2 = composite.shape[1] // 2
    label_params = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(composite, 'GT', (6, h2//10), **label_params)
    cv2.putText(composite, 'Restored', (w2+6, h2//10), **label_params)
    cv2.putText(composite, 'GT mask', (6, h2 + h2//10), **label_params)
    cv2.putText(composite, 'Pred mask', (w2+6, h2 + h2//10), **label_params)

    return composite


def safe_makedirs(p):
    os.makedirs(p, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval', default='results/samples/evaluation.csv', help='Evaluation CSV from evaluate_samples_vs_gt.py')
    p.add_argument('--gt', default='dataset_final_hiragana/gt/test')
    p.add_argument('--restored', default='results/samples')
    p.add_argument('--gt-mask', default='dataset_final_hiragana/mask_random_gt/test')
    p.add_argument('--pred-mask', default='results/samples_pred_masks_binarized_inv')
    p.add_argument('--out', default='results/failures_low_iou')
    p.add_argument('-n', type=int, default=None, help='Number of lowest IoU samples to save (short)')
    p.add_argument('--count', type=int, default=None, help='Number of lowest IoU samples to save (long, avoids shell/conda -n ambiguity)')
    args = p.parse_args()

    eval_csv = args.eval
    if not os.path.isfile(eval_csv):
        raise SystemExit(f'Evaluation CSV not found: {eval_csv}')

    df = pd.read_csv(eval_csv)
    # ensure iou column exists
    if 'iou' not in df.columns:
        raise SystemExit('No iou column in evaluation CSV')

    # select rows with non-null iou, sort ascending
    df_valid = df[df['iou'].notna()].copy()
    df_valid['iou_val'] = pd.to_numeric(df_valid['iou'], errors='coerce')
    df_valid = df_valid.sort_values('iou_val', ascending=True)

    # choose count: prefer --count then -n then default 50
    if args.count is not None:
        n_val = args.count
    elif args.n is not None:
        n_val = args.n
    else:
        n_val = 50
    n = min(n_val, df_valid.shape[0])
    selected = df_valid.head(n)

    out_dir = args.out
    safe_makedirs(out_dir)

    stats = {'iou_mean': float(selected['iou_val'].mean()), 'precision_mean': None, 'recall_mean': None}
    if 'precision' in selected.columns:
        stats['precision_mean'] = float(pd.to_numeric(selected['precision'], errors='coerce').mean())
    if 'recall' in selected.columns:
        stats['recall_mean'] = float(pd.to_numeric(selected['recall'], errors='coerce').mean())

    print(f'Saving {n} lowest-IoU samples to {out_dir}')
    print('Aggregate for selected samples:', stats)

    rows = []
    for idx, row in tqdm(selected.iterrows(), total=n, desc='Saving failures'):
        gt_fn = row['gt_filename']
        stem, _ = os.path.splitext(gt_fn)
        gt_path = find_file_with_exts(args.gt, stem)
        restored_path = find_file_with_exts(args.restored, stem)
        gt_mask_path = find_file_with_exts(args.gt_mask, stem)
        pred_mask_path = find_file_with_exts(args.pred_mask, stem)

        gt = load_img(gt_path)
        restored = load_img(restored_path)
        # load masks as grayscale then to vis
        gt_mask = None
        pred_mask = None
        if gt_mask_path and os.path.isfile(gt_mask_path):
            m = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.max() <= 1:
                    m = (m * 255).astype('uint8')
                gt_mask = mask_to_vis(m)
        if pred_mask_path and os.path.isfile(pred_mask_path):
            m = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.max() <= 1:
                    m = (m * 255).astype('uint8')
                pred_mask = mask_to_vis(m)

        # meta text
        iou = row.get('iou')
        prec = row.get('precision') if 'precision' in row else None
        rec = row.get('recall') if 'recall' in row else None
        meta = f"{stem}  IoU={iou} Prec={prec} Rec={rec}"

        comp = make_composite(gt, restored, gt_mask, pred_mask, meta_text=meta)
        # filename with rank and iou
        rank = len(rows) + 1
        outname = f"{rank:03d}_iou_{float(iou):.3f}_{stem}.png"
        outpath = os.path.join(out_dir, outname)
        cv2.imwrite(outpath, comp)
        rows.append(outpath)

    # Save a simple CSV listing selected files and metrics
    meta_out = os.path.join(out_dir, 'selected_low_iou.csv')
    selected.to_csv(meta_out, index=False)
    print('Wrote selected CSV to', meta_out)
    print('Done.')


if __name__ == '__main__':
    main()
