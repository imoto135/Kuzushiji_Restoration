#!/usr/bin/env python3
"""
MambaIR Image Restoration Script
Restores damaged images using the trained MambaIR Stage 2 model.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add the mamba models directory to the path so we can import mambair
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "models" / "mamba"))
from mambair import MambaIR


def resolve_mask_path(img_path: Path, mask_dir: Path) -> Path:
    """
    Find the corresponding mask file by matching the prefix and removing the damage type suffix.
    Example: type1_sev0.5_U+3042_xxx_Stain.jpg -> mask_dir/U+3042_xxx.png
    """
    damage_types = [
        '_Transparent_Stain', '_Missing', '_Stain', '_Scratch', '_Ghosting',
        '_Abrasion', '_Transparent'
    ]
    
    stem = img_path.stem
    # Remove prefix if present (e.g., type1_sev0.5_)
    import re
    stem = re.sub(r"^type\d+_sev[\d\.]+_", "", stem)
    
    for dt in damage_types:
        if stem.endswith(dt):
            stem = stem[:-len(dt)]
            break
            
    # Check possible mask names
    for suffix in [img_path.suffix, ".png"]:
        candidate1 = mask_dir / f"{stem}{suffix}"
        candidate2 = mask_dir / img_path.name
        
        if candidate1.exists():
            return candidate1
        if candidate2.exists():
            return candidate2
            
    return None


def main():
    parser = argparse.ArgumentParser(description="Image Restoration using MambaIR")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/mamba/experiments/MambaIR_Stage2_v3/best_model.pth",
        help="Path to trained model weights."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/full_padded/lq/test",
        help="Directory containing damaged images."
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="data/full_padded/pred_mask/test",
        help="Directory containing predicted masks."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mamba_restored",
        help="Directory to save restored images."
    )
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--device_id", type=int, default=0, help="GPU ID to use")
    
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up paths
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    if not mask_dir.exists():
        print(f"Error: Mask directory does not exist: {mask_dir}")
        return

    # Load Model
    print(f"Loading MambaIR model from {args.weights}...")
    model = MambaIR(
        upscale=1,
        in_chans=4,
        out_chans=3,
        img_size=args.img_size,
        embed_dim=64,
        depths=(4, 4, 4, 4),
        d_state=16,
    ).to(device)

    try:
        state_dict = torch.load(args.weights, map_location="cpu")
        # Clean up keys if necessary
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()
    print("Model loaded successfully.")

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process.")

    # Process images
    processed_count = 0
    missing_masks = 0

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Restoring images"):
            mask_path = resolve_mask_path(img_path, mask_dir)
            
            if mask_path is None or not mask_path.exists():
                missing_masks += 1
                # Fallback to zero mask if missing
                img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_rgb is None:
                    continue
                h, w = img_rgb.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if img_bgr is None or mask is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Resize if necessary
            if img_rgb.shape[:2] != (args.img_size, args.img_size):
                img_rgb = cv2.resize(img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
            if mask.shape[:2] != (args.img_size, args.img_size):
                mask = cv2.resize(mask, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)

            # Normalize to [0, 1]
            img_normalized = img_rgb.astype(np.float32) / 255.0
            mask_normalized = mask.astype(np.float32) / 255.0
            
            # Convert to Tensor [C, H, W]
            img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))).float()
            mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0).float()
            
            # Concatenate [4, H, W]
            input_tensor = torch.cat([img_tensor, mask_tensor], dim=0).unsqueeze(0).to(device)

            # Inference
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_tensor)
            
            output = outputs.float().clamp(0.0, 1.0)
            
            # Convert back to numpy image
            output_np = output.squeeze(0).cpu().numpy()
            output_np = np.transpose(output_np, (1, 2, 0))  # CHW -> HWC
            output_bgr = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Save
            output_path = output_dir / f"{img_path.stem}_restored.png"
            cv2.imwrite(str(output_path), output_bgr)
            processed_count += 1

    print(f"\nRestoration completed. {processed_count}/{len(image_files)} images saved to {output_dir}")
    if missing_masks > 0:
        print(f"Warning: {missing_masks} images had no corresponding mask file and used an empty mask.")


if __name__ == "__main__":
    main()
