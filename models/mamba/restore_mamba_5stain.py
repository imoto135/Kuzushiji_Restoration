"""
restore_mamba_5stain.py — MambaIR inference for hiragana_fulldataset_5stain

Usage:
  python restore_mamba_5stain.py \
      --mask_type gtmask \
      --weights experiments/MambaIR_5stain_Gtmask/best_model.pth \
      --input_dir ../../data/hiragana_fulldataset_5stain/lq/test \
      --mask_dir  ../../data/hiragana_fulldataset_5stain/gt_mask/test \
      --output_dir ../../outputs/mamba_5stain/gtmask/test \
      --device_id 0
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mambair import MambaIR


# ── ファイルマッチング ──────────────────────────────────────────────────────────
_PREFIX_RE = re.compile(r"^type\d+_sev[\d\.]+_")
_SUFFIX_RE = re.compile(
    r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain|Abrasion|Transparent)$"
)

def _clean_stem(path: Path) -> str:
    stem = path.stem
    stem = _PREFIX_RE.sub("", stem)
    stem = _SUFFIX_RE.sub("", stem)
    return stem

def build_mask_index(mask_dir: Path) -> dict:
    return {
        _clean_stem(f): f
        for f in mask_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


# ── モデルロード ───────────────────────────────────────────────────────────────
def load_model(weights_path: str, in_chans: int, device: torch.device) -> MambaIR:
    model = MambaIR(
        upscale=1, in_chans=in_chans, out_chans=3,
        img_size=128, embed_dim=64, depths=(4, 4, 4, 4), d_state=16,
    )
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    print(f"Loaded weights from {weights_path}  (in_chans={in_chans})")
    return model


# ── 1枚推論 ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def restore_image(
    model: MambaIR,
    img_path: Path,
    mask_path: Path | None,
    device: torch.device,
) -> np.ndarray:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lq = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    if mask_path is not None:
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            mask_gray = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        mask_t = torch.from_numpy(mask_gray).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
        inp = torch.cat([lq, mask_t], dim=1)
    else:
        inp = lq

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp)

    out_np = out.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_type", required=True,
                        choices=["nomask", "predmask", "gtmask"])
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_dir",   default=None,
                        help="predmask/gtmask時に必要")
    parser.add_argument("--device_id",  type=int, default=0)
    args = parser.parse_args()

    device   = torch.device(f"cuda:{args.device_id}")
    in_chans = 3 if args.mask_type == "nomask" else 4

    model = load_model(args.weights, in_chans, device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"Found {len(img_files)} images in {input_dir}")

    mask_index = {}
    if args.mask_type != "nomask":
        if args.mask_dir is None:
            raise ValueError("--mask_dir が必要です (predmask/gtmask)")
        mask_index = build_mask_index(Path(args.mask_dir))

    errors = 0
    for img_path in tqdm(img_files, desc=f"MambaIR [{args.mask_type}]"):
        try:
            mask_path = mask_index.get(_clean_stem(img_path)) if mask_index else None
            restored  = restore_image(model, img_path, mask_path, device)
            out_path  = output_dir / f"{img_path.stem}_restored.png"
            cv2.imwrite(str(out_path), restored)
        except Exception as e:
            print(f"Error: {img_path.name} — {e}")
            errors += 1

    print(f"Done. {len(img_files) - errors}/{len(img_files)} saved to {output_dir}")


if __name__ == "__main__":
    main()
