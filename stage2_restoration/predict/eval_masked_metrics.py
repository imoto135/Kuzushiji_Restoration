#!/usr/bin/env python3
"""
修復結果の評価 (PSNR / SSIM / Masked PSNR / Masked SSIM / LPIPS)

- GT画像、修復画像、二値マスク(0/255 または 0/1)を用いて評価します
- 評価方法:
  - PSNR/SSIM: 画像全体
  - Masked PSNR/Masked SSIM: マスク領域のみ
  - LPIPS: 画像全体

各画像ごとの結果を CSV に保存します。
CSVカラム:
  画像名, 損傷サフィックスの種類, 文字のユニコード, psnr, masked psnr, ssim, masked ssim, lpips
"""
import os
import re
import csv
import math
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import lpips
from skimage.metrics import structural_similarity as sk_ssim

import wandb

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _pil_resample_bicubic():
    try:
        return Image.Resampling.BICUBIC
    except Exception:
        return Image.BICUBIC


def _pil_resample_nearest():
    try:
        return Image.Resampling.NEAREST
    except Exception:
        return Image.NEAREST


def normalize_stem(fname: str) -> str:
    """
    ファイル名のstemを正規化して突合しやすくする。
    座標情報(_X..._Y...)は保持し、指定された損傷サフィックスのみを除去する。
    """
    stem = os.path.splitext(os.path.basename(fname))[0]

    # 1. 指定された損傷サフィックスを削除
    # (_Transparent_Stain を _Stain より先に判定させるため、長い順に記述しています)
    stem = re.sub(r"(_Transparent_Stain|_Ghosting|_Scratch|_Missing|_Stain)$", "", stem, flags=re.IGNORECASE)

    # 2. 推論時に付与されがちなその他のサフィックスも除去
    stem = re.sub(r"(_restored|_pred|_out)$", "", stem, flags=re.IGNORECASE)

    # 3. 末尾に残ったアルファベット1文字などのゴミ（_a, -bなど）があれば除去
    # (座標情報 _Y1234 などを消さないよう、数字を含まないものだけに限定)
    stem = re.sub(r"[_-][A-Za-z]+$", "", stem)

    return stem


def extract_unicode_tag(filename: str) -> str:
    """例: 'U+3042_...' -> 'U+3042'（無ければ空文字）"""
    m = re.search(r"(U\+[0-9A-Fa-f]{4,6})", filename)
    return m.group(1) if m else ""


def extract_damage_suffix(filename: str) -> str:
    """
    損傷サフィックスの種類を抽出。
    例: ..._X0668_Y1844_Transparent_Stain.jpg -> 'Transparent_Stain'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    
    # 1. まず推論時に付く不要なタグ (_restored, _pred など) を除去
    base = re.sub(r"(_restored|_pred|_out)$", "", base, flags=re.IGNORECASE)

    # 2. 座標情報 (_X..._Y...) の後ろにある部分を損傷名として取得
    m = re.search(r"_Y\d+_(.+)$", base)
    if m:
        damage = m.group(1)
        # 末尾に _a, -b のような「1文字の識別子」がついている場合のみ除去
        # (Missingなどを消さないよう、1文字に限定)
        damage = re.sub(r"[_-][a-zA-Z0-9]$", "", damage)
        return damage

    # 座標がない場合などは、末尾の英字部分を損傷名とみなす（保険）
    parts = base.split("_")
    tail = []
    for p in reversed(parts):
        # 明らかな座標やUnicodeタグ以外を拾う
        if "U+" in p or re.match(r"^[XY]\d+$", p):
            break
        tail.append(p)
    
    if tail:
        return "_".join(reversed(tail))
        
    return ""

def read_img_rgb(path: str):
    try:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception as e:
        print(f"[WARN] failed to read rgb: {path} ({e})")
        return None


def read_img_gray(path: str):
    try:
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception as e:
        print(f"[WARN] failed to read gray: {path} ({e})")
        return None


def to_binary_mask(mask_u8: np.ndarray) -> np.ndarray:
    """(H,W) uint8/bool -> float32 {0,1}"""
    if mask_u8.dtype == np.bool_:
        return mask_u8.astype(np.float32)
    return (mask_u8 > 127).astype(np.float32)


def calculate_psnr(img1_u8: np.ndarray, img2_u8: np.ndarray) -> float:
    """全体 PSNR（uint8 RGB, data_range=255）"""
    diff = img1_u8.astype(np.float64) - img2_u8.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def calculate_ssim(img1_u8: np.ndarray, img2_u8: np.ndarray) -> float:
    """全体 SSIM（uint8 RGB, data_range=255）"""
    h, w, _ = img1_u8.shape
    min_side = min(h, w)
    win_size = min(7, min_side if (min_side % 2 == 1) else (min_side - 1))
    win_size = max(3, win_size)

    try:
        v = sk_ssim(img1_u8, img2_u8, win_size=win_size, channel_axis=2, data_range=255)
    except TypeError:
        v = sk_ssim(img1_u8, img2_u8, win_size=win_size, multichannel=True, data_range=255)
    return float(v)


def calculate_masked_psnr(img1_u8: np.ndarray, img2_u8: np.ndarray, mask01: np.ndarray) -> float:
    """
    img1_u8,img2_u8: uint8 (H,W,3)
    mask01: float {0,1} (H,W)  1が評価対象領域
    """
    diff = (img1_u8.astype(np.float64) - img2_u8.astype(np.float64)) / 255.0
    m = mask01[:, :, None]
    denom = (np.sum(m) * 3.0 + 1e-8)
    mse = float(np.sum((diff ** 2) * m) / denom)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def calculate_masked_ssim(img1_u8: np.ndarray, img2_u8: np.ndarray, mask01: np.ndarray) -> float:
    """
    skimage の SSIM マップ(full=True)を取り、マスク領域のみ平均。
    """
    h, w, _ = img1_u8.shape
    min_side = min(h, w)
    win_size = min(7, min_side if (min_side % 2 == 1) else (min_side - 1))
    win_size = max(3, win_size)

    try:
        _, ssim_map = sk_ssim(
            img1_u8, img2_u8,
            win_size=win_size,
            channel_axis=2,
            data_range=255,
            full=True
        )
    except TypeError:
        _, ssim_map = sk_ssim(
            img1_u8, img2_u8,
            win_size=win_size,
            multichannel=True,
            data_range=255,
            full=True
        )

    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=2)
    masked = float(np.sum(ssim_map * mask01) / (np.sum(mask01) + 1e-8))
    return masked


def build_index(dir_path: str):
    """
    dir 内の画像を stem 正規化で index 化:
      normalized_stem -> filepath
    同一キーが複数ある場合は最初のものを採用。
    """
    m = {}
    p = Path(dir_path)
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            key = normalize_stem(f.name)
            if key not in m:
                m[key] = str(f)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--restored-dir", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--cpu", action="store_true")

    # CSV
    ap.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="各画像の評価結果CSVの出力先（未指定なら restored-dir/eval_results.csv）"
    )

    # WandB
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="Kuzushiji_Restoration")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-name", default="eval_masked_metrics")
    ap.add_argument("--max-log-images", type=int, default=50)
    args = ap.parse_args()

    csv_out = args.csv_out or os.path.join(args.restored_dir, "eval_results.csv")

    use_wandb = bool(args.use_wandb and (not args.no_wandb))
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )
        table = wandb.Table(columns=[
            "id", "gt", "restored", "mask",
            "psnr", "masked_psnr", "ssim", "masked_ssim", "lpips"
        ])
    else:
        table = None

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    print(f"[INFO] device: {device}")

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    gt_map = build_index(args.gt_dir)
    res_map = build_index(args.restored_dir)
    mask_map = build_index(args.mask_dir)

    print("\n[DEBUG] GTの例:      ", list(gt_map.keys())[:1])
    print("[DEBUG] Restoredの例:", list(res_map.keys())[:1])
    print("[DEBUG] Maskの例:    ", list(mask_map.keys())[:1], "\n")

    keys = sorted(set(gt_map.keys()) & set(res_map.keys()) & set(mask_map.keys()))
    if not keys:
        raise SystemExit("[ERROR] No matched files among gt/restored/mask (after normalize_stem).")

    # 集計用
    total_psnr = 0.0
    total_mpsnr = 0.0
    total_ssim = 0.0
    total_mssim = 0.0
    total_lpips = 0.0
    count = 0

    # CSV行
    rows = []
    header = ["画像名", "損傷サフィックスの種類", "文字のユニコード", "psnr", "masked psnr", "ssim", "masked ssim", "lpips"]

    for i, k in enumerate(tqdm(keys, desc="Eval")):
        gt_path = gt_map[k]
        rs_path = res_map[k]
        mk_path = mask_map[k]

        gt = read_img_rgb(gt_path)
        rs = read_img_rgb(rs_path)
        mk = read_img_gray(mk_path)
        if gt is None or rs is None or mk is None:
            continue

        # サイズ合わせ（GT基準）
        h, w, _ = gt.shape
        if rs.shape != gt.shape:
            rs = np.array(Image.fromarray(rs).resize((w, h), _pil_resample_bicubic()), dtype=np.uint8)
        if mk.shape != (h, w):
            mk = np.array(Image.fromarray(mk).resize((w, h), _pil_resample_nearest()), dtype=np.uint8)

        mask01 = to_binary_mask(mk)

        psnr_v = calculate_psnr(gt, rs)
        ssim_v = calculate_ssim(gt, rs)
        mpsnr_v = calculate_masked_psnr(gt, rs, mask01)
        mssim_v = calculate_masked_ssim(gt, rs, mask01)

        # LPIPS（全体）
        t_gt = torch.from_numpy(gt.transpose(2, 0, 1)).float() / 127.5 - 1.0
        t_rs = torch.from_numpy(rs.transpose(2, 0, 1)).float() / 127.5 - 1.0
        t_gt = t_gt.unsqueeze(0).to(device)
        t_rs = t_rs.unsqueeze(0).to(device)
        with torch.no_grad():
            lpips_v = float(loss_fn(t_gt, t_rs).item())

        total_psnr += psnr_v
        total_ssim += ssim_v
        total_mpsnr += mpsnr_v
        total_mssim += mssim_v
        total_lpips += lpips_v
        count += 1

        # CSV: 画像名は restored のファイル名を採用（損傷サフィックス抽出のため）
        img_name = os.path.basename(rs_path)
        damage = extract_damage_suffix(img_name)
        uni = extract_unicode_tag(img_name)

        rows.append([
            img_name,
            damage,
            uni,
            psnr_v,
            mpsnr_v,
            ssim_v,
            mssim_v,
            lpips_v
        ])

        if use_wandb and i < args.max_log_images:
            table.add_data(
                k,
                wandb.Image(gt),
                wandb.Image(rs),
                wandb.Image((mask01 * 255).astype(np.uint8)),
                psnr_v, mpsnr_v, ssim_v, mssim_v, lpips_v
            )

    # 平均
    avg_psnr = total_psnr / count if count else 0.0
    avg_ssim = total_ssim / count if count else 0.0
    avg_mpsnr = total_mpsnr / count if count else 0.0
    avg_mssim = total_mssim / count if count else 0.0
    avg_lpips = total_lpips / count if count else 0.0

    print("==== Results ====")
    print(f"count         : {count}")
    print(f"psnr          : {avg_psnr:.6f}")
    print(f"masked_psnr   : {avg_mpsnr:.6f}")
    print(f"ssim          : {avg_ssim:.6f}")
    print(f"masked_ssim   : {avg_mssim:.6f}")
    print(f"lpips         : {avg_lpips:.6f}")

    # CSV 保存（各画像の結果）
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[INFO] wrote csv: {csv_out}")

    if use_wandb:
        wandb.log(
            {
                "count": count,
                "avg_psnr": avg_psnr,
                "avg_masked_psnr": avg_mpsnr,
                "avg_ssim": avg_ssim,
                "avg_masked_ssim": avg_mssim,
                "avg_lpips": avg_lpips,
                "evaluation_table": table,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()