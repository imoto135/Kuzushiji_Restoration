#!/usr/bin/env python3
"""
Binarize restored images in `results/samples` to produce predicted masks for evaluation.
Saves masks to `results/samples_pred_masks_binarized` by default, preserving basenames.

Options:
 - --thresh : threshold (0-255) for binarization (default 127)
 - --morph : apply morphological opening/closing (True/False)
 - --min-area : remove connected components smaller than this (pixels)
"""
import os
import argparse
import cv2
import numpy as np
import glob


def binarize_image(img_path, thresh=127, morph=False, min_area=0):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # normalize if needed
    if img.max() > 1:
        _, bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    else:
        # assume 0/1 image
        bw = (img > 0).astype('uint8') * 255
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    if min_area > 0:
        # remove small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((bw>0).astype('uint8'), connectivity=8)
        out = np.zeros_like(bw)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                out[labels == i] = 255
        bw = out
    return bw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restored-dir', default='results/samples', help='Directory with restored images')
    parser.add_argument('--out-dir', default='results/samples_pred_masks_binarized', help='Where to save predicted masks')
    parser.add_argument('--thresh', type=int, default=127)
    parser.add_argument('--morph', action='store_true')
    parser.add_argument('--min-area', type=int, default=0)
    args = parser.parse_args()

    restored_dir = args.restored_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([p for p in glob.glob(os.path.join(restored_dir, '*')) if os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg')])
    if not files:
        print('No restored images found in', restored_dir)
        return

    for p in files:
        basename = os.path.splitext(os.path.basename(p))[0]
        bw = binarize_image(p, thresh=args.thresh, morph=args.morph, min_area=args.min_area)
        if bw is None:
            print('Skipping', p)
            continue
        out_path = os.path.join(out_dir, basename + '.png')
        cv2.imwrite(out_path, bw)
        print('Saved', out_path)

    print('Done. Masks saved to', out_dir)

if __name__ == '__main__':
    main()
