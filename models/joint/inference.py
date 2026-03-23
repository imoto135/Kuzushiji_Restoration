"""
inference.py — Joint UNet++/NAFNet inference

Usage:
  cd /home/imoto/Kuzushiji_Restoration/models/joint
  python inference.py \\
      --model_path experiments/joint_unetpp_nafnet_charb_percep/best_model.pth \\
      --input_dir  ../../data/hiragana_fulldataset_5stain/lq/test \\
      --output_dir ../../outputs/joint_test
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

_THIS   = Path(__file__).resolve().parent
_MODELS = _THIS.parent
_NAFNET = _MODELS / "nafnet"
for _p in [str(_NAFNET)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint_model import JointRestorationNet


def load_image(path: Path) -> torch.Tensor:
    """Load image as (1, 3, H, W) float32 tensor in [0, 1]."""
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save (1, 3, H, W) or (3, H, W) float32 tensor in [0,1] as PNG."""
    t = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((t * 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint model inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pth (joint checkpoint)")
    parser.add_argument("--input_dir",  type=str, required=True,
                        help="Directory of degraded LQ images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save restored images")
    parser.add_argument("--save_mask",  action="store_true",
                        help="Also save predicted damage masks")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir   = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    model = JointRestorationNet(unetpp_pretrain=None, nafnet_pretrain=None).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded joint model from {args.model_path}")

    # ── Inference ─────────────────────────────────────────────────────────
    img_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Processing {len(img_files)} images …")

    with torch.no_grad():
        for img_path in tqdm(img_files):
            lq = load_image(img_path).to(device)

            with torch.cuda.amp.autocast():
                restored, mask, _temp = model(lq)

            # Save restored image (same name, PNG)
            out_name = img_path.stem + ".png"
            save_image(restored, output_dir / out_name)

            if args.save_mask:
                # mask: (1, 1, H, W) → grayscale PNG
                m = mask.squeeze().clamp(0, 1).cpu().numpy()
                Image.fromarray((m * 255).astype(np.uint8)).save(mask_dir / out_name)

    print(f"Done. Saved to {output_dir}")


if __name__ == "__main__":
    main()
