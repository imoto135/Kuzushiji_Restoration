#!/usr/bin/env python3
"""
同じ文字（ファイル名に含まれる U+xxxx のプレフィックス）を横並びに5枚並べて
results/compare_all に保存する。

使用法（プロジェクトルートで）:
python scripts/compare_same_characters.py
"""
import os
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 入力ディレクトリ（順番が出力の並び順になる）
GT_DIR = "dataset_final_hiragana/gt/test"
LQ_DIR = "dataset_final_hiragana/lq_random/test"
R_WGT_DIR = "results/restormer_withgtmask/restored"
R_NET140K_DIR = "results/restored_net_g_140000/restored"
R_NOMASK2_DIR = "results/restormer_nomask2/restored"

OUT_DIR = "results/compare_all"
os.makedirs(OUT_DIR, exist_ok=True)

# 画像出力サイズ基準（高さ）。各画像はアスペクト維持でこの高さにリサイズされる
TARGET_H = 256
JPEG_QUALITY = 95

# 正規表現で文字コード (例 U+3042) を取り出す
code_re = re.compile(r'(U\+\w+)')

def list_files_map(d):
    files = sorted([p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    # map: stem -> Path
    m = {p.stem: p for p in files}
    # also keep list for searching
    return m, files

def extract_code(stem):
    m = code_re.search(stem)
    if m:
        return m.group(1)
    # fallback: prefix before first underscore
    return stem.split('_')[0]

def find_best_match(target_stem, code, dir_map, dir_list):
    # 1) exact stem
    if target_stem in dir_map:
        return dir_map[target_stem]
    # 2) any file whose stem startswith target_stem
    for s, p in dir_map.items():
        if s.startswith(target_stem):
            return p
    # 3) any file whose stem contains the code
    for p in dir_list:
        if code in p.stem:
            return p
    # none
    return None

def make_missing_image(w, h, text="MISSING"):
    img = Image.new("RGB", (w, h), (255,255,255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    tw, th = draw.textsize(text, font=font)
    draw.text(((w-tw)//2, (h-th)//2), text, fill=(200,0,0), font=font)
    return img

# prepare maps
gt_map, gt_list = list_files_map(GT_DIR)
lq_map, lq_list = list_files_map(LQ_DIR)
r_wgt_map, r_wgt_list = list_files_map(R_WGT_DIR)
r_140_map, r_140_list = list_files_map(R_NET140K_DIR)
r_nomask_map, r_nomask_list = list_files_map(R_NOMASK2_DIR)

# iterate GT files to keep instances consistent
for gt_path in tqdm(gt_list, desc="Making comparisons"):
    gt_stem = gt_path.stem
    code = extract_code(gt_stem)

    # find matching files in other dirs (prefer same stem / instance)
    lq_p = find_best_match(gt_stem, code, lq_map, lq_list)
    r_wgt_p = find_best_match(gt_stem, code, r_wgt_map, r_wgt_list)
    r_140_p = find_best_match(gt_stem, code, r_140_map, r_140_list)
    r_nomask_p = find_best_match(gt_stem, code, r_nomask_map, r_nomask_list)

    cols = [gt_path, lq_p, r_wgt_p, r_140_p, r_nomask_p]
    imgs = []
    # load images, resize to common height
    for p in cols:
        if p is None:
            imgs.append(None)
            continue
        try:
            im = Image.open(p).convert("RGB")
            # resize preserving aspect ratio to TARGET_H
            w, h = im.size
            if h != TARGET_H:
                nw = max(1, int(w * (TARGET_H / h)))
                im = im.resize((nw, TARGET_H), Image.BICUBIC)
            imgs.append(im)
        except Exception:
            imgs.append(None)

    # compute widths: if all None skip
    if all(i is None for i in imgs):
        continue

    # compute total width
    widths = [im.size[0] if im is not None else TARGET_H for im in imgs]
    total_w = sum(widths)
    out = Image.new("RGB", (total_w, TARGET_H), (255,255,255))
    x = 0
    labels = ["GT","LQ","Restormer_wGTmask","Restored_net_g_140000","Restormer_nomask2"]
    for im, wlabel in zip(imgs, labels):
        if im is None:
            # make placeholder of width TARGET_H (square) or average width
            ph = TARGET_H
            pw = TARGET_H
            placeholder = make_missing_image(pw, ph, text="MISSING")
            out.paste(placeholder, (x,0))
            # draw label
            x += pw
        else:
            out.paste(im, (x,0))
            x += im.size[0]

    # save with filename: code + gt_stem (to be unique)
    safe_name = gt_stem
    out_path = Path(OUT_DIR) / f"{safe_name}.jpg"
    out.save(out_path, format='JPEG', quality=95)
# done
print("Saved comparisons to", OUT_DIR)