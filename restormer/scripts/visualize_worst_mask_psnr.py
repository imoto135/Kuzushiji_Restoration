#!/usr/bin/env python3
import os, csv, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm

GT = "dataset_final_hiragana/gt/test"
PRED = "results/restormer_withmask/restored"
MASK = "dataset_final_hiragana/mask_gt/test"
ANALYSIS_CSV = "analysis.csv"  # analyze_mask_psnr の出力
OUTDIR = "results/analysis_vis"
N = 20

os.makedirs(OUTDIR, exist_ok=True)

rows = []
with open(ANALYSIS_CSV, newline='', encoding='utf-8') as f:
    r = csv.reader(f)
    hdr = next(r)
    for row in r:
        if not row: continue
        fn = row[0]
        try:
            mask_ratio = float(row[1])
            masked_psnr = float(row[2]) if row[2] != '' else float('nan')
            full_psnr = float(row[3]) if row[3] != '' else float('nan')
        except:
            continue
        rows.append((fn, mask_ratio, masked_psnr, full_psnr))

rows = sorted(rows, key=lambda x: (np.inf if np.isnan(x[2]) else x[2]))
for idx, (fn, mr, mpsnr, fpsnr) in enumerate(rows[:N]):
    gt = cv2.imread(os.path.join(GT, fn))
    pred = cv2.imread(os.path.join(PRED, Path(fn).with_suffix('.png').name))
    mask = cv2.imread(os.path.join(MASK, fn), cv2.IMREAD_GRAYSCALE)
    if gt is None or pred is None or mask is None:
        continue
    # resize pred/mask to GT if needed
    if pred.shape[:2] != gt.shape[:2]:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != gt.shape[:2]:
        mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    diff = cv2.absdiff(gt, pred)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(np.uint8(np.clip(diff_gray*4,0,255)), cv2.COLORMAP_JET)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    comp = np.concatenate([gt, mask_color, pred, heat], axis=1)
    cv2.putText(comp, f"{fn} mpsnr={mpsnr:.2f} fullpsnr={fpsnr:.2f} mask_r={mr:.3f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    outp = os.path.join(OUTDIR, f"{idx:02d}_{Path(fn).stem}.png")
    cv2.imwrite(outp, comp)
print("Saved visualizations to", OUTDIR)