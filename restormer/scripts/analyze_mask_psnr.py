#!/usr/bin/env python3
import os
import csv
import numpy as np
import cv2
from tqdm import tqdm
from scipy import stats

# reuse function from calculate_metrics.py if available, otherwise duplicate minimal PSNR
try:
    from Kuzushiji_Restoration.scripts.calculate_metrics import calculate_masked_metrics
except Exception:
    def calculate_masked_metrics(img_true, img_test, mask):
        if img_true.shape != img_test.shape:
            h, w = img_true.shape[:2]
            img_test = cv2.resize(img_test, (w, h))
        if mask.shape[:2] != img_true.shape[:2]:
            h, w = img_true.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = (mask > 128)
        if mask_bool.sum() == 0:
            return float('nan'), float('nan')
        true_f = img_true.astype(np.float32)
        test_f = img_test.astype(np.float32)
        true_pixels = true_f[mask_bool]
        test_pixels = test_f[mask_bool]
        mse = np.mean((true_pixels - test_pixels) ** 2)
        if mse == 0 or np.isnan(mse):
            psnr = 100.0
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        # simple ssim omitted for speed; return nan
        return psnr, float('nan')

def full_image_psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    if a.shape != b.shape:
        h, w = a.shape[:2]
        b = cv2.resize(b, (w, h))
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main(gt_dir, pred_dir, mask_dir, out_csv='analysis_mask_psnr.csv'):
    # robust file pairing by stem (ignore extensions / handle suffixes)
    def list_imgs(d):
        return sorted([f for f in os.listdir(d) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    gt_files = list_imgs(gt_dir)
    pred_files = list_imgs(pred_dir)
    mask_files = list_imgs(mask_dir)

    def make_map(files):
        m = {}
        for f in files:
            stem = os.path.splitext(f)[0]
            m.setdefault(stem, []).append(f)
        return m

    pred_map = make_map(pred_files)
    mask_map = make_map(mask_files)

    rows = []
    not_found_pred = 0
    not_found_mask = 0
    for fn in tqdm(gt_files):
        p_gt = os.path.join(gt_dir, fn)
        gt_stem = os.path.splitext(fn)[0]

        # find prediction file by exact stem or by best prefix match
        pred_fname = None
        if gt_stem in pred_map:
            pred_fname = pred_map[gt_stem][0]
        else:
            # try pred stems that startwith gt_stem or vice versa (pick longest match)
            best = None; best_len = 0
            for ps in pred_map.keys():
                if ps.startswith(gt_stem) and len(ps) > best_len:
                    best = ps; best_len = len(ps)
                if gt_stem.startswith(ps) and len(ps) > best_len:
                    best = ps; best_len = len(ps)
            if best:
                pred_fname = pred_map[best][0]

        if pred_fname is None:
            not_found_pred += 1
            continue

        # find mask similarly
        mask_fname = None
        if gt_stem in mask_map:
            mask_fname = mask_map[gt_stem][0]
        else:
            best = None; best_len = 0
            for ms in mask_map.keys():
                if ms.startswith(gt_stem) and len(ms) > best_len:
                    best = ms; best_len = len(ms)
                if gt_stem.startswith(ms) and len(ms) > best_len:
                    best = ms; best_len = len(ms)
            if best:
                mask_fname = mask_map[best][0]

        if mask_fname is None:
            not_found_mask += 1
            continue

        try:
            gt = cv2.imread(p_gt)
            pred = cv2.imread(os.path.join(pred_dir, pred_fname))
            mask = cv2.imread(os.path.join(mask_dir, mask_fname), cv2.IMREAD_GRAYSCALE)
            mask_bool = (mask > 128)
            mask_ratio = mask_bool.sum() / (mask.shape[0] * mask.shape[1])
            masked_psnr, _ = calculate_masked_metrics(gt, pred, mask)
            full_psnr = full_image_psnr(gt, pred)
            rows.append((fn, mask_ratio, masked_psnr, full_psnr))
        except Exception as e:
            print('Error', fn, e)
            continue

    print(f"GT total: {len(gt_files)}, paired: {len(rows)}, missing_pred: {not_found_pred}, missing_mask: {not_found_mask}")

    # save CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filename','mask_ratio','masked_psnr','full_psnr'])
        w.writerows(rows)

    # summary (safe extraction per-column, handle empty / NaN)
    if len(rows) == 0:
        print('No valid rows collected.')
        return
    mask_ratios = np.array([float(r[1]) if r[1] is not None else np.nan for r in rows], dtype=float)
    masked_psnrs = np.array([float(r[2]) if r[2] is not None else np.nan for r in rows], dtype=float)
    full_psnrs = np.array([float(r[3]) if r[3] is not None else np.nan for r in rows], dtype=float)

    print('count:', len(rows))
    print('mask_ratio mean/median/std:', np.nanmean(mask_ratios), np.nanmedian(mask_ratios), np.nanstd(mask_ratios))
    print('masked_psnr mean/median/std:', np.nanmean(masked_psnrs), np.nanmedian(masked_psnrs), np.nanstd(masked_psnrs))
    print('full_psnr mean/median/std:', np.nanmean(full_psnrs), np.nanmedian(full_psnrs), np.nanstd(full_psnrs))

    # correlation
    valid = ~np.isnan(masked_psnrs)
    if valid.sum() > 10:
        r, p = stats.pearsonr(mask_ratios[valid], masked_psnrs[valid])
        print(f'pearson(mask_ratio, masked_psnr) r={r:.4f} p={p:.4e}')
        r2, p2 = stats.pearsonr(full_psnrs[valid], masked_psnrs[valid])
        print(f'pearson(full_psnr, masked_psnr) r={r2:.4f} p={p2:.4e}')

    # print worst examples by masked_psnr
    rows_sorted = sorted(rows, key=lambda x: (np.inf if np.isnan(x[2]) else x[2]))
    worst = rows_sorted[:20]
    print('\nWorst 20 by masked_psnr:')
    for fn, mr, mpsnr, fpsnr in worst[:20]:
        print(f'{fn} mask_ratio={mr:.4f} masked_psnr={mpsnr:.3f} full_psnr={fpsnr:.3f}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gt', default='dataset_final_hiragana/gt/test')
    p.add_argument('--pred', default='results/restored_net_g_140000')
    p.add_argument('--mask', default='dataset_final_hiragana/mask_gt/test')
    p.add_argument('--out', default='analysis_mask_psnr.csv')
    args = p.parse_args()
    main(args.gt, args.pred, args.mask, args.out)