#!/usr/bin/env python3
"""
マスクなし（LQのみ）で Restormer により修復画像を生成して保存するスクリプト。

例:
python stage2_restoration/predict/infer_restormer_nomask.py \
  --weights /path/to/model.pth \
  --data-dir datasets/hiragana_dataset \
  --lq-dir lq/test \
  --out-dir outputs/restored_nomask_test \
  --image-size 128 \
  --batch-size 8
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# --- import 解決（restormer/basicsr を見つける）---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESTORMER_PKG_DIR = os.path.join(PROJECT_ROOT, 'restormer')
for p in (PROJECT_ROOT, RESTORMER_PKG_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from restormer.basicsr.models.archs.restormer_arch import Restormer  # noqa

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _pil_resample_bicubic():
    # Pillow 9/10 両対応
    try:
        return Image.Resampling.BICUBIC
    except Exception:
        return Image.BICUBIC


class LQDataset(Dataset):
    def __init__(self, lq_dir: str, image_size: int):
        self.lq_dir = lq_dir
        self.image_size = image_size

        files = []
        for p in sorted(Path(lq_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p.name)
        self.files = files

        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found in lq_dir: {lq_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        lq_path = os.path.join(self.lq_dir, name)

        lq = Image.open(lq_path).convert("RGB")
        if self.image_size and self.image_size > 0:
            lq = lq.resize((self.image_size, self.image_size), _pil_resample_bicubic())

        lq_np = np.array(lq, dtype=np.float32) / 255.0  # HWC, [0,1]
        x = torch.from_numpy(lq_np.transpose(2, 0, 1))  # 3HW
        return x, name


def build_model(inp_channels: int = 3, out_channels: int = 3):
    # infer_restormer_with_mask.py に倣い、同じ Restormer の基本設定を使用
    return Restormer(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "params_ema", "best_model"):
            if k in ckpt:
                return ckpt[k]
    return ckpt


def strip_module(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", default="datasets/hiragana_dataset")
    ap.add_argument("--lq-dir", default="lq/test")
    ap.add_argument("--out-dir", default="outputs/restored_nomask_test")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save-ext", default=".png", choices=[".png", ".jpg"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    logging.info(f"Using device: {device}")

    lq_dir = args.lq_dir if os.path.isabs(args.lq_dir) else os.path.join(args.data_dir, args.lq_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = LQDataset(lq_dir, args.image_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = build_model(inp_channels=3, out_channels=3).to(device).eval()

    ckpt = torch.load(args.weights, map_location=device)
    sd = unwrap_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict-like mapping.")
    sd = strip_module(sd)

    try:
        model.load_state_dict(sd, strict=True)
        logging.info("Weights loaded (strict=True).")
    except Exception as e:
        logging.warning(f"Strict load failed: {e}. Retrying strict=False.")
        model.load_state_dict(sd, strict=False)
        logging.info("Weights loaded (strict=False).")

    use_amp = bool(args.use_amp and device.type == "cuda")
    saved = 0

    with torch.no_grad():
        for x, names in tqdm(dl, desc="Inference"):
            x = x.to(device, dtype=torch.float32)

            if use_amp:
                with torch.cuda.amp.autocast():
                    y = model(x)
            else:
                y = model(x)

            # 実装差分吸収（list/tuple で返る場合）
            if isinstance(y, (list, tuple)):
                y = y[0]

            y = torch.clamp(y, 0.0, 1.0).cpu().numpy()  # BCHW

            for i, name in enumerate(names):
                img = (y[i].transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
                out_name = str(Path(name).with_suffix(args.save_ext))
                out_path = os.path.join(args.out_dir, out_name)
                Image.fromarray(img).save(out_path)
                saved += 1

    logging.info(f"Saved {saved} images to: {args.out_dir}")


if __name__ == "__main__":
    main()