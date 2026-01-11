#!/usr/bin/env python3
"""
Create side-by-side sample comparison images for quick visual inspection.
Saves images to `results/samples_compare/`.
Columns: GT | Extracted(Restored rightmost) | Restored composite | Pred Mask (color) | GT Mask (color)
"""
import os
import cv2
import glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESTORED_DIR = os.path.join(ROOT, 'results', 'restored_net_g_140000')
GT_DIR = os.path.join(ROOT, 'dataset_final_hiragana', 'gt', 'test')
PRED_MASK_DIR = os.path.join(ROOT, 'dataset_final_hiragana', 'mask_random_prediction', 'test')
GT_MASK_DIR = os.path.join(ROOT, 'dataset_final_hiragana', 'mask_random_gt', 'test')
OUT_DIR = os.path.join(ROOT, 'results', 'samples_compare')
N = 8

os.makedirs(OUT_DIR, exist_ok=True)

# build mapping of stems -> filenames for GT and masks
def stem(fp):
    return os.path.splitext(os.path.basename(fp))[0]

gt_files = {stem(p): p for p in glob.glob(os.path.join(GT_DIR, '*'))}
pred_mask_files = {stem(p): p for p in glob.glob(os.path.join(PRED_MASK_DIR, '*'))}
gt_mask_files = {stem(p): p for p in glob.glob(os.path.join(GT_MASK_DIR, '*'))}

# list restored images
restored_paths = sorted(glob.glob(os.path.join(RESTORED_DIR, '*')))
# filter common image extensions
restored_paths = [p for p in restored_paths if os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp')]

def find_matching_gt(restored_name):
    s = stem(restored_name)
    # try exact
    if s in gt_files:
        return gt_files[s]
    # try removing common suffixes
    for suf in ['_prediction', '_pred', '_restored', '_recon', '_out']:
        if s.endswith(suf):
            key = s[:-len(suf)]
            if key in gt_files:
                return gt_files[key]
    # try substring match: find gt whose stem is contained in restored stem
    for k,v in gt_files.items():
        if k in s or s in k:
            return v
    return None


def find_mask_by_stem_map(s, mapping):
    # try exact
    if s in mapping:
        return mapping[s]
    for suf in ['_prediction', '_mask', '_pred']:
        if s.endswith(suf):
            key = s[:-len(suf)]
            if key in mapping:
                return mapping[key]
    # substring
    for k,v in mapping.items():
        if k in s or s in k:
            return v
    return None


def colorize_mask(mask_gray, target_size=None):
    if target_size is not None:
        mask_gray = cv2.resize(mask_gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    if len(mask_gray.shape) == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    colored = cv2.applyColorMap(b, cv2.COLORMAP_JET)
    return colored

saved = []
count = 0
for p in restored_paths:
    if count >= N:
        break
    gt_path = find_matching_gt(p)
    if gt_path is None:
        # skip if no gt match
        continue
    # read images
    restored = cv2.imread(p, cv2.IMREAD_COLOR)
    gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    if restored is None or gt is None:
        continue
    gt_h, gt_w = gt.shape[:2]
    # extract rightmost tile matching GT width if possible
    if restored.shape[1] >= gt_w:
        restored_extracted = restored[:, -gt_w:]
    else:
        # fallback: center crop to gt_w
        pad = gt_w - restored.shape[1]
        restored_extracted = cv2.copyMakeBorder(restored, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
        restored_extracted = restored_extracted[:, -gt_w:]
    # read masks
    s = stem(p)
    pred_mask_path = find_mask_by_stem_map(s, pred_mask_files)
    gt_mask_path = find_mask_by_stem_map(s, gt_mask_files)
    pred_mask_col = np.zeros_like(restored_extracted)
    gt_mask_col = np.zeros_like(restored_extracted)
    if pred_mask_path and os.path.exists(pred_mask_path):
        pm = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        pred_mask_col = colorize_mask(pm, target_size=gt.shape[:2])
    if gt_mask_path and os.path.exists(gt_mask_path):
        gm = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_col = colorize_mask(gm, target_size=gt.shape[:2])

    # also resize restored_extracted to GT size if needed
    if restored_extracted.shape[:2] != gt.shape[:2]:
        restored_extracted = cv2.resize(restored_extracted, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)

    # ensure composite is reasonable size: resize to gt size for display
    restored_comp_display = cv2.resize(restored, (gt_w, gt_h), interpolation=cv2.INTER_AREA)

    # assemble horizontally: GT | Restored_extracted | Restored_composite | Pred_mask | GT_mask
    pieces = [gt, restored_extracted, restored_comp_display, pred_mask_col, gt_mask_col]
    # if any piece is None or zero shape, replace with blank
    final_pieces = []
    for pc in pieces:
        if pc is None or pc.size == 0:
            final_pieces.append(np.zeros((gt_h, gt_w, 3), dtype=np.uint8))
        else:
            if pc.shape[0] != gt_h or pc.shape[1] != gt_w:
                pc = cv2.resize(pc, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
            final_pieces.append(pc)

    out = np.hstack(final_pieces)
    out_name = os.path.join(OUT_DIR, f"sample_{count+1:03d}_{stem(p)}.png")
    cv2.imwrite(out_name, out)
    print('Saved', out_name)
    saved.append(out_name)
    count += 1

print('\nDone. Saved {} sample images to {}'.format(len(saved), OUT_DIR))
