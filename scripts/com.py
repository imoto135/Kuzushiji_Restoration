#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 入力ディレクトリ（左からこの順で横並びになります）
DIRS = [
    "dataset_final_hiragana/lq_random/test",
    "dataset_final_hiragana/gt/test",
    "dataset_final_hiragana/mask_gt/test",
    "results/unet_nomask/masks",
    "results/deeplabv3p_mask",
    "dataset_final_hiragana/mask_random_prediction/test"
]
OUT_DIR = "results/lastcompare"
TARGET_H = 128  # 出力高さ（各画像はアスペクト比を保ってリサイズ）
os.makedirs(OUT_DIR, exist_ok=True)

# 正規化関数: サフィックス（_alphaNN, _prediction 等）を除去して比較
_suffix_re = re.compile(r'(_alpha\d+|_prediction|_pred|_restored|_mask|_gt)$', re.IGNORECASE)
def normalize(stem: str) -> str:
    return _suffix_re.sub('', stem)

def list_files_map(d: str):
    p = Path(d)
    files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png')]
    # return list of (stem, path)
    return [(f.stem, f) for f in files]

def find_best_match(target_stem: str, candidates):
    norm_target = normalize(target_stem)
    # 1) exact stem
    for stem, p in candidates:
        if stem == target_stem:
            return p
    # 2) exact normalized stem
    for stem, p in candidates:
        if normalize(stem) == norm_target:
            return p
    # 3) contains normalized target
    for stem, p in candidates:
        if norm_target in normalize(stem):
            return p
    # 4) longest common prefix
    best = None
    best_len = 0
    for stem, p in candidates:
        lcp = len(os.path.commonprefix([normalize(stem), norm_target]))
        if lcp > best_len:
            best_len = lcp
            best = p
    return best

def make_placeholder(width, height):
    return Image.new("RGB", (width, height), (255,255,255))

def concat_for_target(target_path: str):
    target = Path(target_path)
    if not target.exists():
        print("Target file not found:", target_path)
        return
    target_stem = target.stem

    maps = [list_files_map(d) for d in DIRS]

    found_imgs = []
    # For width reference, open target image first
    try:
        tgt_img = Image.open(target).convert("RGB")
    except Exception as e:
        print("Failed to open target:", e)
        return
    # Find match in each dir (prefer same file if dir equals target's dir)
    for d, cand in zip(DIRS, maps):
        # if target is inside this dir, use it directly
        cand_path = None
        if Path(d) == target.parent:
            cand_path = target
        else:
            match = find_best_match(target_stem, cand)
            if match:
                cand_path = match
        if cand_path is None:
            found_imgs.append(None)
        else:
            try:
                im = Image.open(cand_path).convert("RGB")
                found_imgs.append(im)
            except Exception:
                found_imgs.append(None)

    # ensure we have at least one image (target itself)
    # prepare resized images (preserve aspect)
    resized = []
    for im in found_imgs:
        if im is None:
            resized.append(None)
            continue
        w,h = im.size
        nh = TARGET_H
        nw = max(1, int(w * (nh / h)))
        resized.append(im.resize((nw, nh), Image.BICUBIC))

    # if all None skip
    if all(im is None for im in resized):
        print("No images found for target in any directory.")
        return

    # replace None with placeholder of average width of available images or target-derived width
    avail_widths = [im.size[0] for im in resized if im is not None]
    if not avail_widths:
        avg_w = TARGET_H
    else:
        avg_w = int(sum(avail_widths) / len(avail_widths))
    for i, im in enumerate(resized):
        if im is None:
            resized[i] = make_placeholder(avg_w, TARGET_H)

    total_w = sum(im.size[0] for im in resized)
    out = Image.new("RGB", (total_w, TARGET_H), (255,255,255))
    x = 0
    for im in resized:
        out.paste(im, (x,0))
        x += im.size[0]

    out_name = f"{target_stem}.jpg"
    out_path = Path(OUT_DIR) / out_name
    out.save(out_path, format='JPEG', quality=95)
    print("Saved:", out_path)

def main():
    parser = argparse.ArgumentParser(description="Concat same-image matches horizontally")
    parser.add_argument('target', help='target image path to find matches for')
    parser.add_argument('--height', type=int, default=128, help='output height for each tile')
    args = parser.parse_args()
    global TARGET_H
    TARGET_H = args.height
    concat_for_target(args.target)

if __name__ == '__main__':
    main()