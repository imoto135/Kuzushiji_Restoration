#!/usr/bin/env python3
"""Invert predicted mask images (0<->255) and save to output dir.

Usage:
  python scripts/invert_pred_masks.py --in-dir results/samples_pred_masks_binarized --out-dir results/samples_pred_masks_binarized_inv [--in-place]

If --in-place is set, files in the input dir will be overwritten (be careful).
"""
import argparse
import os
from glob import glob
import cv2
from tqdm import tqdm


def invert_image(img):
    # img: numpy array
    import numpy as np
    if img is None:
        return None
    # If has alpha channel, preserve it
    if img.ndim == 2:
        return 255 - img
    if img.ndim == 3:
        h, w, c = img.shape
        if c == 1:
            return 255 - img
        if c == 3:
            return 255 - img
        if c == 4:
            rgb = img[..., :3]
            a = img[..., 3]
            rgb_inv = 255 - rgb
            out = cv2.merge([rgb_inv[...,0], rgb_inv[...,1], rgb_inv[...,2], a])
            return out
    # fallback
    return 255 - img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--in-place', action='store_true', help='Overwrite input files (be careful)')
    args = p.parse_args()

    indir = args.in_dir
    outdir = args.out_dir
    in_place = args.in_place

    if not os.path.isdir(indir):
        raise SystemExit(f"Input directory not found: {indir}")
    if in_place:
        outdir = indir
    else:
        os.makedirs(outdir, exist_ok=True)

    patterns = ['**/*.png', '**/*.PNG', '**/*.jpg', '**/*.jpeg']
    files = []
    for pat in patterns:
        files.extend(glob(os.path.join(indir, pat), recursive=True))

    files = sorted(files)
    if not files:
        print('No mask files found in', indir)
        return

    for f in tqdm(files, desc='Inverting masks'):
        rel = os.path.relpath(f, indir)
        outp = os.path.join(outdir, rel)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print('Failed to read', f)
            continue
        inv = invert_image(img)
        # Ensure binary-like output: map non-zero to 255, zero stays 0 if input not pure binary
        # But preserve as 0/255 values after inversion
        # If image has multiple channels, we keep as-is
        # Save with same extension
        ext = os.path.splitext(f)[1].lower()
        # For grayscale or single-channel, force PNG single-channel
        if inv.ndim == 2:
            save_flag = []
            cv2.imwrite(outp, inv)
        else:
            cv2.imwrite(outp, inv)

    print('Done. Inverted masks saved to', outdir)


if __name__ == '__main__':
    main()
