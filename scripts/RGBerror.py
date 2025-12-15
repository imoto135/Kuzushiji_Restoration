#!/usr/bin/env python3
import os
import re
import argparse
import logging
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def _as_float01_mask(p: Path) -> np.ndarray:
    """Load grayscale mask image as float32 in [0,1], shape (H,W)."""
    arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _stem(p: Path) -> str:
    return p.stem


def _clean_stem(stem: str) -> str:
    """
    GT と soft の対応付け用に stem を正規化。
    - 末尾の損傷種サフィックスを除去
    - 末尾の _lq/_input/_Missing などを除去
    - もし "U+XXXX_typeN_sevX.Y_" のような形なら type/sev 部分だけ落として U+XXXX は保持
    """
    # 例: U+3042_type1_sev0.3_xxx_Scratch -> U+3042_xxx_Scratch
    stem = re.sub(r"^(U\+[0-9A-Fa-f]{4,6}_)?type\d+_sev[\d\.]+_", r"\1", stem)

    # 末尾の損傷種を除去
    stem = re.sub(r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain)$", "", stem, flags=re.IGNORECASE)

    # よくある末尾サフィックスを除去
    stem = re.sub(r"(_Missing|_lq|_input)$", "", stem, flags=re.IGNORECASE)

    # 念のため "_" 連続を潰す
    stem = re.sub(r"__+", "_", stem).strip("_")
    return stem


def _parse_meta_from_stem(stem: str) -> dict:
    """
    ファイル名(stem)からメタ情報を抽出:
      - codepoint: U+3042 など（先頭優先、なければ全体検索）
      - char: 可能なら対応する実文字（例: あ）
      - severity: sev0.3 等があれば抽出（なければ None）
      - damage: Scratch 等（stem内で最後にマッチしたものを採用）
    """
    # codepoint
    m_cp = re.match(r"^(U\+[0-9A-Fa-f]{4,6})", stem)
    if m_cp is None:
        m_cp = re.search(r"(U\+[0-9A-Fa-f]{4,6})", stem)
    codepoint = m_cp.group(1) if m_cp else None

    # char
    ch = None
    if codepoint:
        try:
            ch = chr(int(codepoint[2:], 16))
        except Exception:
            ch = None

    # severity
    m_sev = re.search(r"(?:^|_)sev(\d+(?:\.\d+)?)(?:_|$)", stem, flags=re.IGNORECASE)
    severity = float(m_sev.group(1)) if m_sev else None

    # damage（最後に出現したものを採用）
    damage = "Unknown"
    dmg_pat = re.compile(r"(?:^|_)(Ghosting|Missing|Stain|Scratch|Transparent_Stain)(?:_|$)", re.IGNORECASE)
    last = None
    for m in dmg_pat.finditer(stem):
        last = m
    if last is not None:
        # 正規化（候補の表記に揃える）
        d = last.group(1)
        # Transparent_Stain だけは大文字小文字混在しやすいので明示的に揃える
        if d.lower() == "transparent_stain":
            damage = "Transparent_Stain"
        else:
            damage = d[0].upper() + d[1:].lower()

    return {
        "codepoint": codepoint,
        "char": ch,
        "severity": severity,
        "damage": damage,
    }


def match_pairs(gt_dir: Path, soft_dir: Path):
    gt_files = [p for p in gt_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    soft_files = [p for p in soft_dir.iterdir() if p.is_file() and not p.name.startswith(".")]

    # soft側を clean_stem -> Path に（同一キーが複数あれば先勝ち）
    soft_map = {}
    for p in soft_files:
        key = _clean_stem(_stem(p))
        soft_map.setdefault(key, p)

    pairs = []
    for gt in sorted(gt_files):
        key = _clean_stem(_stem(gt))
        sp = soft_map.get(key)
        if sp is not None:
            pairs.append((gt, sp))
    return pairs


def visualize_comparison(input_img, soft_pred, gt_mask):
    """
    input_img: (H, W, 3)
    soft_pred: (H, W) 0.0-1.0
    gt_mask:   (H, W) 0 or 1
    """
    hard_pred = (soft_pred > 0.5).astype(np.float32)
    diff_map = np.abs(gt_mask - soft_pred)

    error_vis = np.zeros((*gt_mask.shape, 3))
    error_vis[..., 0] = soft_pred
    error_vis[..., 1] = gt_mask

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1); plt.title("Input")
    plt.imshow(input_img)

    plt.subplot(1, 5, 2); plt.title("Soft Prediction (Heatmap)")
    plt.imshow(soft_pred, cmap='jet', vmin=0, vmax=1)

    plt.subplot(1, 5, 3); plt.title("GT (Hard)")
    plt.imshow(gt_mask, cmap='gray')

    plt.subplot(1, 5, 4); plt.title("Difference (|GT - Soft|)")
    plt.imshow(diff_map, cmap='magma')

    plt.subplot(1, 5, 5); plt.title("Error Analysis (R:Pred, G:GT)")
    plt.imshow(error_vis)

    plt.show()


def visualize_comparison_rgb(soft_pred01: np.ndarray, gt01: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    RGBエラーマップ:
      - Green: TP
      - Red: FP
      - Blue: FN
    """
    soft = np.clip(soft_pred01.astype(np.float32), 0.0, 1.0)
    gt = (gt01 > 0.5).astype(np.float32)

    hard = (soft > thr).astype(np.float32)

    tp = (gt == 1) & (hard == 1)
    fp = (gt == 0) & (hard == 1)
    fn = (gt == 1) & (hard == 0)

    R = np.zeros_like(soft, dtype=np.float32)
    G = np.zeros_like(soft, dtype=np.float32)
    B = np.zeros_like(soft, dtype=np.float32)

    R[fp] = soft[fp]
    G[tp] = soft[tp]
    B[fn] = (1.0 - soft[fn])

    bg = (0.12 * soft).astype(np.float32)
    R = np.clip(R + bg, 0.0, 1.0)
    G = np.clip(G + bg, 0.0, 1.0)
    B = np.clip(B + bg, 0.0, 1.0)

    rgb = (np.stack([R, G, B], axis=-1) * 255.0).round().astype(np.uint8)
    return rgb


def compute_basic_metrics(soft_pred01: np.ndarray, gt01: np.ndarray, thr: float = 0.5, eps: float = 1e-7):
    soft = np.clip(soft_pred01.astype(np.float32), 0.0, 1.0)
    gt = (gt01 > 0.5).astype(np.float32)
    pred = (soft > thr).astype(np.float32)

    inter = float((pred * gt).sum())
    union = float(pred.sum() + gt.sum() - inter)
    iou = (inter + eps) / (union + eps)

    precision = (inter + eps) / (float(pred.sum()) + eps)
    recall = (inter + eps) / (float(gt.sum()) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    mae = float(np.abs(soft - gt).mean())

    # confusion (pixel counts)
    tp = float(((pred == 1) & (gt == 1)).sum())
    fp = float(((pred == 1) & (gt == 0)).sum())
    fn = float(((pred == 0) & (gt == 1)).sum())
    tn = float(((pred == 0) & (gt == 0)).sum())

    return {
        "IoU": float(iou),
        "F1": float(f1),
        "MAE": float(mae),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": float(precision),
        "recall": float(recall),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare hard GT masks and soft predicted masks; save RGB error visualization + CSV.")
    p.add_argument("--gt-dir", type=str,
                   default="Kuzushiji_Restoration/datasets/dataset_final_hiragana/mask_gt/test")
    p.add_argument("--soft-dir", type=str,
                   default="Kuzushiji_Restoration/datasets/dataset_final_hiragana/mask_prediction_dropout/test")
    p.add_argument("--out-dir", type=str,
                   default="Kuzushiji_Restoration/experiments/compare")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--ext", type=str, default=".png", help="output extension: .png or .jpg")
    p.add_argument("--max-items", type=int, default=0, help="0 means all")

    # 【追加】CSV出力
    p.add_argument("--csv-path", type=str, default="", help="per-image metrics csv (default: <out-dir>/per_image_metrics.csv)")
    p.add_argument("--summary-csv-path", type=str, default="", help="group summary csv (default: <out-dir>/summary_by_codepoint_damage.csv)")
    return p.parse_args()


def _write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    gt_dir = Path(args.gt_dir)
    soft_dir = Path(args.soft_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_path) if args.csv_path else (out_dir / "per_image_metrics.csv")
    summary_csv_path = Path(args.summary_csv_path) if args.summary_csv_path else (out_dir / "summary_by_codepoint_damage.csv")

    pairs = match_pairs(gt_dir, soft_dir)
    if not pairs:
        logging.error("No pairs found. Check dirs:\n  gt=%s\n  soft=%s", gt_dir, soft_dir)
        return

    if args.max_items and args.max_items > 0:
        pairs = pairs[: args.max_items]

    logging.info("Pairs: %d", len(pairs))

    per_image_rows = []
    # group summary accumulator: (codepoint, damage) -> stats
    group = {}  # key -> {"n":..., "IoU_sum":..., ...}

    for gt_path, soft_path in pairs:
        gt = _as_float01_mask(gt_path)
        soft = _as_float01_mask(soft_path)

        if soft.shape != gt.shape:
            soft_img = Image.open(soft_path).convert("L").resize((gt.shape[1], gt.shape[0]), resample=Image.BILINEAR)
            soft = np.array(soft_img, dtype=np.float32) / 255.0

        rgb = visualize_comparison_rgb(soft, gt, thr=args.thr)

        metrics = compute_basic_metrics(soft, gt, thr=args.thr)

        # 【修正】まずGTから抽出し、damage等が取れない場合はsoftから補完
        meta = _parse_meta_from_stem(gt_path.stem)
        if meta["damage"] == "Unknown":
            meta_soft = _parse_meta_from_stem(soft_path.stem)
            meta["damage"] = meta_soft["damage"]
        if meta["severity"] is None:
            meta_soft = _parse_meta_from_stem(soft_path.stem)
            meta["severity"] = meta_soft["severity"]

        out_name = f"{gt_path.stem}{args.ext}"
        Image.fromarray(rgb).save(out_dir / out_name)

        row = {
            "id_clean": _clean_stem(gt_path.stem),
            "gt_stem": gt_path.stem,
            "gt_file": gt_path.name,
            "soft_file": soft_path.name,
            "codepoint": meta["codepoint"],
            "char": meta["char"],
            "severity": meta["severity"],
            "damage": meta["damage"],
            "thr": args.thr,
            "H": int(gt.shape[0]),
            "W": int(gt.shape[1]),
            "rgb_vis_file": out_name,
            **metrics,
        }
        per_image_rows.append(row)

        gkey = (meta["codepoint"], meta["damage"])
        if gkey not in group:
            group[gkey] = {"n": 0, "IoU_sum": 0.0, "F1_sum": 0.0, "MAE_sum": 0.0}
        group[gkey]["n"] += 1
        group[gkey]["IoU_sum"] += row["IoU"]
        group[gkey]["F1_sum"] += row["F1"]
        group[gkey]["MAE_sum"] += row["MAE"]

    # per-image csv
    fieldnames = [
        "id_clean", "gt_stem", "gt_file", "soft_file",
        "codepoint", "char", "severity", "damage",
        "thr", "H", "W", "rgb_vis_file",
        "IoU", "F1", "MAE", "precision", "recall", "TP", "FP", "FN", "TN",
    ]
    _write_csv(csv_path, per_image_rows, fieldnames)
    logging.info("Saved per-image CSV: %s", csv_path)

    # summary csv (codepoint x damage)
    summary_rows = []
    for (cp, damage), acc in sorted(group.items(), key=lambda x: (x[0][0] is None, str(x[0][0]), x[0][1])):
        n = acc["n"]
        summary_rows.append({
            "codepoint": cp,
            "damage": damage,
            "n": n,
            "IoU_mean": acc["IoU_sum"] / n if n else None,
            "F1_mean": acc["F1_sum"] / n if n else None,
            "MAE_mean": acc["MAE_sum"] / n if n else None,
        })
    _write_csv(summary_csv_path, summary_rows, ["codepoint", "damage", "n", "IoU_mean", "F1_mean", "MAE_mean"])
    logging.info("Saved summary CSV: %s", summary_csv_path)

    # 全体平均ログ
    IoU_mean = float(np.mean([r["IoU"] for r in per_image_rows]))
    F1_mean = float(np.mean([r["F1"] for r in per_image_rows]))
    MAE_mean = float(np.mean([r["MAE"] for r in per_image_rows]))
    logging.info("Done. Saved RGB to: %s", out_dir)
    logging.info("Overall mean (thr=%.2f): IoU=%.4f, F1=%.4f, MAE=%.6f", args.thr, IoU_mean, F1_mean, MAE_mean)


if __name__ == "__main__":
    main()