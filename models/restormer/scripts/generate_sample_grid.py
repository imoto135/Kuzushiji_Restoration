#!/usr/bin/env python3
import os
import argparse
import random
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

def find_file_for_stem(d, stem):
    if not d:
        return None
    exts = ['.png', '.jpg', '.jpeg']
    # exact
    for e in exts:
        p = os.path.join(d, stem + e)
        if os.path.isfile(p):
            return p
    # common suffixes
    candidates = [stem, stem + '_prediction', stem + '_mask', stem + '_pred']
    for c in candidates:
        for e in exts:
            p = os.path.join(d, c + e)
            if os.path.isfile(p):
                return p
    # fallback: any file in d that startswith stem
    for p in glob(os.path.join(d, stem + '*')):
        if os.path.isfile(p):
            return p
    # partial match search
    for p in glob(os.path.join(d, '*')):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.startswith(stem) or stem.startswith(name):
            return p
    return None

def load_rgb(path, target_size):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        # placeholder blank
        im = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
    return im

def load_mask_vis(path, target_size):
    m = None
    if path and os.path.isfile(path):
        mm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mm is not None:
            mm = cv2.resize(mm, target_size, interpolation=cv2.INTER_NEAREST)
            # make white foreground (assume >127 is foreground)
            fg = (mm > 127).astype(np.uint8) * 255
            m = np.stack([fg, fg, fg], axis=2)
    if m is None:
        m = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    return m

def put_label(img, text, height=24):
    # add top padding for text
    h, w = img.shape[:2]
    pad = np.zeros((height, w, 3), dtype=np.uint8) + 255
    cv2.putText(pad, text, (6, height-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return np.vstack([pad, img])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', default='dataset_final_hiragana/gt/test')
    parser.add_argument('--lq-dir', default='dataset_final_hiragana/lq_random/test')
    parser.add_argument('--gt-mask-dir', default='dataset_final_hiragana/mask_gt/test')
    parser.add_argument('--pred-mask-dir', default='dataset_final_hiragana/mask_random_prediction/test')
    parser.add_argument('--restored-dir', default='results/samples')
    parser.add_argument('--out-dir', default='results/samples_compare_grid')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=512, help='resize short side to this (square output)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt_files = sorted([os.path.basename(p) for p in glob(os.path.join(args.gt_dir, '*')) if os.path.isfile(p)])
    # get stems
    stems = [os.path.splitext(n)[0] for n in gt_files]
    random.seed(args.seed)
    random.shuffle(stems)

    selected = []
    for s in stems:
        # ensure at least restored or pred-mask exists; still include even if some pieces missing
        selected.append(s)
        if len(selected) >= args.num_samples:
            break

    target_size = (args.size, args.size)

    for idx, stem in enumerate(tqdm(selected, desc='Generating samples')):
        gt_img_p = find_file_for_stem(args.gt_dir, stem)
        lq_img_p = find_file_for_stem(args.lq_dir, stem)
        gt_mask_p = find_file_for_stem(args.gt_mask_dir, stem)
        pred_mask_p = find_file_for_stem(args.pred_mask_dir, stem)
        restored_p = find_file_for_stem(args.restored_dir, stem)

        gt_img = load_rgb(gt_img_p, target_size)
        lq_img = load_rgb(lq_img_p, target_size)
        gt_mask = load_mask_vis(gt_mask_p, target_size)
        pred_mask = load_mask_vis(pred_mask_p, target_size)
        restored_img = load_rgb(restored_p, target_size)

        # label rows
        cols = [
            ('GT', gt_img),
            ('Damaged', lq_img),
            ('GT mask', gt_mask),
            ('Pred mask', pred_mask),
            ('Restored', restored_img),
        ]
        # add labels
        labeled = [put_label(img, label) for label, img in cols]
        # ensure same height after label
        heights = [im.shape[0] for im in labeled]
        H = max(heights)
        # pad to same height
        padded = []
        for im in labeled:
            h, w = im.shape[:2]
            if h < H:
                pad = np.ones((H - h, w, 3), dtype=np.uint8) * 255
                im = np.vstack([im, pad])
            padded.append(im)
        # concatenate horizontally
        composite = np.hstack(padded)

        out_name = f"{idx+1:03d}_{stem}.png"
        out_path = os.path.join(args.out_dir, out_name)
        # convert RGB -> BGR for cv2.imwrite
        cv2.imwrite(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    print("Saved", min(len(selected), args.num_samples), "samples to", args.out_dir)

if __name__ == '__main__':
    main()