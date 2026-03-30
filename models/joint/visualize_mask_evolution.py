import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_THIS = Path(__file__).resolve().parent
_MODELS = _THIS.parent
_NAFNET = _MODELS / "nafnet"
for _p in [str(_NAFNET)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint_model import JointRestorationNet

def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))

def to_tensor(img_np: np.ndarray) -> torch.Tensor:
    img = img_np.astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Visualize GT Mask vs Predicted Soft Mask")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--image_name", type=str, required=True, help="Filename of the image (e.g., sample_01.png)")
    parser.add_argument("--base_dir", type=str, default="../../data/full_padded", help="Base data directory")
    parser.add_argument("--split", type=str, default="val", help="train, val, or test")
    parser.add_argument("--output", type=str, default="mask_comparison.png", help="Output visualization plot path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base_dir)
    
    # Construct paths
    lq_path = base / "lq" / args.split / args.image_name
    gt_path = base / "gt" / args.split / args.image_name
    gt_mask_path = base / "gt_mask" / args.split / args.image_name

    for p in [lq_path, gt_path, gt_mask_path]:
        if not p.exists():
            print(f"Error: {p} does not exist!")
            sys.exit(1)

    # Load data
    lq_np = load_image(lq_path)
    gt_np = load_image(gt_path)
    gt_mask_np = load_mask(gt_mask_path)
    
    lq_tensor = to_tensor(lq_np).to(device)

    # Load Model
    model = JointRestorationNet(unetpp_pretrain=None, nafnet_pretrain=None).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # Inference
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            restored_tensor, mask_tensor, temp_tau = model(lq_tensor)
            
    restored_np = restored_tensor.squeeze().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    pred_mask_np = mask_tensor.squeeze().clamp(0, 1).cpu().numpy()

    # Visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    axes[0].imshow(lq_np)
    axes[0].set_title("Input (Degraded)")
    axes[0].axis("off")

    axes[1].imshow(gt_mask_np, cmap="gray")
    axes[1].set_title("GT Mask (Binary Label)")
    axes[1].axis("off")

    axes[2].imshow(pred_mask_np, cmap="gray")
    axes[2].set_title(f"Predicted Soft Mask (τ={temp_tau:.2f})")
    axes[2].axis("off")

    axes[3].imshow(restored_np)
    axes[3].set_title("Restored Output")
    axes[3].axis("off")

    axes[4].imshow(gt_np)
    axes[4].set_title("Ground Truth (Clean)")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    main()
