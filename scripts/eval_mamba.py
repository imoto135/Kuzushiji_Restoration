#!/usr/bin/env python3
"""
MambaIR Stage2 Evaluation Script
- PSNR / SSIM / LPIPS をテストセットで計算
- results/ に結果を保存
"""

import argparse
import math
import re
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# skimage not required — SSIM computed inline
from tqdm import tqdm

# MambaIR のモジュールへのパス
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "models" / "mamba"))
from mambair import MambaIR  # noqa: E402

# ──────────────────────────────────────────────
# Dataset helpers (train_mamba.py と同じロジック)
# ──────────────────────────────────────────────
_PREFIX_RE = re.compile(r"^type\d+_sev[\d\.]+ _")
_SUFFIX_RE = re.compile(
    r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain|Abrasion|Transparent)$"
)


def _clean_stem(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^type\d+_sev[\d\.]+_", "", stem)
    stem = re.sub(
        r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain|Abrasion|Transparent)$",
        "",
        stem,
    )
    return stem


def _index_dir(directory: Path):
    return {
        _clean_stem(f): f
        for f in directory.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def build_triplets(lq_dir, gt_dir, mask_dir):
    lq_idx = _index_dir(lq_dir)
    gt_idx = _index_dir(gt_dir)
    mask_idx = _index_dir(mask_dir)
    common = set(lq_idx) & set(gt_idx) & set(mask_idx)
    return sorted([(lq_idx[k], gt_idx[k], mask_idx[k]) for k in common])


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def calculate_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: float32 [0,1] HWC"""
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: float32 [0,1] HWC — simplified SSIM (luminance+contrast+structure)"""
    C1, C2 = (0.01 ** 2), (0.03 ** 2)
    # flatten to vectors per channel then average
    ssim_vals = []
    for c in range(pred.shape[2]):
        p = pred[:, :, c].astype(np.float64)
        g = gt[:, :, c].astype(np.float64)
        mu_p, mu_g = p.mean(), g.mean()
        sig_p = p.std()
        sig_g = g.std()
        sig_pg = np.mean((p - mu_p) * (g - mu_g))
        ssim_c = ((2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)) / \
                 ((mu_p**2 + mu_g**2 + C1) * (sig_p**2 + sig_g**2 + C2))
        ssim_vals.append(ssim_c)
    return float(np.mean(ssim_vals))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate MambaIR on test set")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/mamba/experiments/MambaIR_Stage2_v3/best_model.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--lq_dir",
        type=str,
        default="data/full_padded/lq/test",
        help="LQ (damaged) test images",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="data/full_padded/gt/test",
        help="GT (clean) test images",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="data/full_padded/pred_mask/test",
        help="Predicted mask directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mamba_eval",
        help="Directory to save restored images (set '' to skip saving)",
    )
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--device_id", type=int, default=1, help="GPU ID")
    parser.add_argument(
        "--save_images", action="store_true", help="Save restored images"
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ──────────────────────────────
    model = MambaIR(
        upscale=1,
        in_chans=4,
        out_chans=3,
        img_size=args.img_size,
        embed_dim=64,
        depths=(4, 4, 4, 4),
        d_state=16,
    ).to(device)

    weights_path = Path(args.weights)
    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    # strip possible "module." prefix
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded.")

    # ── LPIPS ───────────────────────────────────
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # ── Dataset ─────────────────────────────────
    lq_dir = Path(args.lq_dir)
    gt_dir = Path(args.gt_dir)
    mask_dir = Path(args.mask_dir)

    triplets = build_triplets(lq_dir, gt_dir, mask_dir)
    print(f"Test samples: {len(triplets)}")

    if args.save_images:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate ────────────────────────────────
    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for lq_p, gt_p, mask_p in tqdm(triplets, desc="Evaluating"):
            # Load
            lq_img = np.array(Image.open(lq_p).convert("RGB").resize(
                (args.img_size, args.img_size), Image.BICUBIC
            )).astype(np.float32) / 255.0
            gt_img = np.array(Image.open(gt_p).convert("RGB").resize(
                (args.img_size, args.img_size), Image.BICUBIC
            )).astype(np.float32) / 255.0
            mask_img = np.array(Image.open(mask_p).convert("L").resize(
                (args.img_size, args.img_size), Image.NEAREST
            )).astype(np.float32) / 255.0

            # To tensor [C, H, W]
            lq_t = torch.from_numpy(lq_img.transpose(2, 0, 1)).float()
            gt_t = torch.from_numpy(gt_img.transpose(2, 0, 1)).float()
            mask_t = torch.from_numpy(mask_img).unsqueeze(0).float()

            inp = torch.cat([lq_t, mask_t], dim=0).unsqueeze(0).to(device)  # [1, 4, H, W]
            gt_batch = gt_t.unsqueeze(0).to(device)  # [1, 3, H, W]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(inp)
            pred = pred.float().clamp(0.0, 1.0)

            # PSNR / SSIM (numpy)
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr = calculate_psnr(pred_np, gt_img)
            s = calculate_ssim(pred_np, gt_img)
            psnr_list.append(psnr)
            ssim_list.append(s)

            # LPIPS ([-1, 1] range)
            lp = lpips_fn(pred * 2 - 1, gt_batch * 2 - 1).item()
            lpips_list.append(lp)

            # Save if requested
            if args.save_images:
                out_path = out_dir / f"{lq_p.stem}_restored.png"
                Image.fromarray((pred_np * 255).astype(np.uint8)).save(out_path)

    # ── Results ─────────────────────────────────
    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    mean_lpips = np.mean(lpips_list)

    print("\n" + "=" * 50)
    print(f"  Model   : {args.weights}")
    print(f"  Samples : {len(triplets)}")
    print(f"  PSNR    : {mean_psnr:.4f} dB")
    print(f"  SSIM    : {mean_ssim:.4f}")
    print(f"  LPIPS   : {mean_lpips:.4f}")
    print("=" * 50)

    # Save CSV
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    exp_name = weights_path.parent.name
    csv_path = results_dir / f"eval_mamba_{exp_name}.csv"
    with open(csv_path, "w") as f:
        f.write("model,psnr,ssim,lpips,n_samples\n")
        f.write(f"{exp_name},{mean_psnr:.4f},{mean_ssim:.4f},{mean_lpips:.4f},{len(triplets)}\n")
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
