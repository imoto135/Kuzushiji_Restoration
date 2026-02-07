#!/usr/bin/env python3
"""
Apply a trained Restormer to LQ images using precomputed character masks.

Features:
- Auto-find latest checkpoint in `experiments/UNet_Restormer_Hiragana_Run01/models` if none provided
- Robustly load checkpoint (handles keys: 'params','state_dict','model', or raw state_dict)
- Save restored images (side-by-side with GT/LQ if available) and CSV of metrics (PSNR/SSIM/IoU)

Usage example:
  python scripts/apply_restormer_with_masks.py --checkpoint experiments/UNet_Restormer_Hiragana_Run01/models/net_g_15000.pth

Defaults assume this repository layout; override with CLI args as needed.
"""
import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# import Restormer architecture from the project
from basicsr.models.archs.restormer_arch import Restormer


def find_latest_checkpoint(models_dir):
    if not os.path.isdir(models_dir):
        return None
    files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not files:
        return None
    # sort by numeric part if present, else by mtime
    def keyfn(f):
        name = os.path.splitext(f)[0]
        nums = ''.join([c for c in name if c.isdigit()])
        if nums:
            return int(nums)
        return os.path.getmtime(os.path.join(models_dir, f))
    files = sorted(files, key=keyfn)
    return os.path.join(models_dir, files[-1])


def load_checkpoint_to_model(model, path, device):
    ck = torch.load(path, map_location='cpu')
    # possible keys
    state = None
    for k in ('params', 'state_dict', 'model', 'net_g'):
        if isinstance(ck, dict) and k in ck:
            state = ck[k]
            break
    if state is None and isinstance(ck, dict):
        # maybe the dict itself is the state_dict
        state = ck
    if state is None:
        raise RuntimeError(f'Unable to find model state in checkpoint: {path}')

    # Some saved dicts have 'module.' prefixes from DataParallel
    new_state = {}
    for k, v in state.items():
        new_k = k
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        new_state[new_k] = v

    model.load_state_dict(new_state)
    model.to(device)
    model.eval()


def img_to_tensor_rgb(img_pil):
    # returns tensor (C,H,W) float32 in [0,1]
    tensor = transforms.ToTensor()(img_pil)
    return tensor


def img_to_tensor_mask(mask_pil):
    # convert to single channel float tensor [0,1]
    mask = np.array(mask_pil.convert('L'), dtype=np.float32) / 255.0
    mask = np.expand_dims(mask, 0)
    return torch.from_numpy(mask)


def find_mask_file_for(fn, mask_dir):
    """Robustly find a mask file for a given filename under mask_dir.

    Tries (in order):
      - exact filename in mask_dir
      - same basename with common suffixes ('', '_prediction', '_mask') and extensions
      - look under mask_dir/predicted_masks
      - limited recursive search for files with matching basename (no heavy IO)
    Returns absolute path or None.
    """
    stem = os.path.splitext(fn)[0]
    exts = ('.jpg', '.png', '.jpeg')

    # 1) direct
    cand = os.path.join(mask_dir, fn)
    if os.path.isfile(cand):
        return cand

    # 2) try suffix variants and extensions
    suffixes = ['', '_prediction', '_mask']
    for suf in suffixes:
        for ext in exts:
            cand = os.path.join(mask_dir, stem + suf + ext)
            if os.path.isfile(cand):
                return cand

    # 3) try predicted_masks subfolder
    pred_sub = os.path.join(mask_dir, 'predicted_masks')
    if os.path.isdir(pred_sub):
        cand = os.path.join(pred_sub, fn)
        if os.path.isfile(cand):
            return cand
        for suf in suffixes:
            for ext in exts:
                cand = os.path.join(pred_sub, stem + suf + ext)
                if os.path.isfile(cand):
                    return cand

    # 4) limited recursive search (depth-first) but stop early when found
    # This helps when masks are stored in nested folders or have different prefixes
    try:
        for root, dirs, files in os.walk(mask_dir):
            # keep this loop light: only check filenames that share the stem
            for f in files:
                name_stem = os.path.splitext(f)[0]
                if name_stem == stem or name_stem.startswith(stem) or stem.startswith(name_stem):
                    return os.path.join(root, f)
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to restormer checkpoint (.pth). If omitted, auto-find latest in experiments/.../models')
    parser.add_argument('--models-dir', type=str, default='experiments/UNet_Restormer_Hiragana_Run01/models', help='Models dir to auto-search')
    parser.add_argument('--input-lq', type=str, default='dataset_final_hiragana/lq_random/test', help='LQ input images')
    parser.add_argument('--input-gt', type=str, default='dataset_final_hiragana/gt/test', help='GT images (optional, used for metrics)')
    parser.add_argument('--mask-root', type=str, default='dataset_final_hiragana/mask_random_prediction', help='Root folder containing predicted masks (subfolders: train/val/test)')
    parser.add_argument('--mask-subset', type=str, default='test', help='Which subset inside mask root to use (train/val/test)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (defaults to results/restored_<checkpoint_basename>)')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu; default auto')
    parser.add_argument('--save-comparison', action='store_true', help='Save side-by-side comparison (GT | LQ | Restored). If GT missing, saves LQ | Restored')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    if args.checkpoint is None:
        ck = find_latest_checkpoint(args.models_dir)
        if ck is None:
            raise SystemExit('No checkpoint found; provide --checkpoint or ensure models dir exists')
        checkpoint_path = ck
    else:
        checkpoint_path = args.checkpoint

    if args.output_dir is None:
        base = os.path.splitext(os.path.basename(checkpoint_path))[0]
        args.output_dir = os.path.join('results', f'restored_{base}')

    os.makedirs(args.output_dir, exist_ok=True)

    # build restormer with typical defaults used in this repo
    restormer = Restormer(
        inp_channels=4, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
    )

    print(f'Loading checkpoint: {checkpoint_path}')
    load_checkpoint_to_model(restormer, checkpoint_path, device)

    input_lq_dir = args.input_lq
    input_gt_dir = args.input_gt if os.path.isdir(args.input_gt) else None
    mask_dir = os.path.join(args.mask_root, args.mask_subset)
    if not os.path.isdir(mask_dir):
        raise SystemExit(f'Mask dir not found: {mask_dir}')

    files = sorted([f for f in os.listdir(input_lq_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        raise SystemExit(f'No input images found in {input_lq_dir}')

    results = []
    to_tensor = transforms.ToTensor()

    for fname in tqdm(files, desc='Restoring'):
        lq_path = os.path.join(input_lq_dir, fname)
        mask_path = find_mask_file_for(fname, mask_dir)
        if mask_path is None:
            print(f'Warning: mask not found for {fname}, skipping')
            continue

        img_lq = Image.open(lq_path).convert('RGB')
        img_mask = Image.open(mask_path).convert('L')

        t_lq = img_to_tensor_rgb(img_lq)
        t_mask = img_to_tensor_mask(img_mask)
        inp = torch.cat([t_lq, t_mask], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = restormer(inp)
            out_t = out[0] if isinstance(out, tuple) else out
            out_t = torch.clamp(out_t, 0.0, 1.0).cpu()

        restored_rgb = (out_t.squeeze().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

        lq_bgr = cv2.cvtColor(np.array(img_lq), cv2.COLOR_RGB2BGR)

        gt_bgr = None
        if input_gt_dir:
            gt_path = os.path.join(input_gt_dir, fname)
            if os.path.isfile(gt_path):
                gt_bgr = cv2.imread(gt_path)

        # metrics: PSNR, SSIM, IoU, Dice, Precision, Recall, Accuracy
        psnr_v, ssim_v = None, None
        iou_v = dice_v = precision_v = recall_v = accuracy_v = None
        if gt_bgr is not None:
            try:
                psnr_v = cv2.PSNR(gt_bgr, restored_bgr)
            except Exception:
                psnr_v = None
            try:
                ssim_v = ssim(gt_bgr, restored_bgr, channel_axis=-1, data_range=255)
            except Exception:
                ssim_v = None

            try:
                gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
                restored_gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
                _, gt_mask_bin = cv2.threshold(gt_gray, 10, 1, cv2.THRESH_BINARY)
                _, restored_mask_bin = cv2.threshold(restored_gray, 10, 1, cv2.THRESH_BINARY)

                gt_bin = (gt_mask_bin > 0).astype(np.uint8)
                pred_bin = (restored_mask_bin > 0).astype(np.uint8)

                tp = int(np.logical_and(gt_bin == 1, pred_bin == 1).sum())
                tn = int(np.logical_and(gt_bin == 0, pred_bin == 0).sum())
                fp = int(np.logical_and(gt_bin == 0, pred_bin == 1).sum())
                fn_ = int(np.logical_and(gt_bin == 1, pred_bin == 0).sum())

                union = tp + fp + fn_
                iou_v = (tp / union) if union > 0 else 0.0

                denom = (2 * tp + fp + fn_)
                dice_v = (2 * tp / denom) if denom > 0 else 0.0

                prec_denom = (tp + fp)
                precision_v = (tp / prec_denom) if prec_denom > 0 else 0.0

                recall_denom = (tp + fn_)
                recall_v = (tp / recall_denom) if recall_denom > 0 else 0.0

                total = tp + tn + fp + fn_
                accuracy_v = ((tp + tn) / total) if total > 0 else 0.0
            except Exception:
                iou_v = dice_v = precision_v = recall_v = accuracy_v = None

        results.append([fname, psnr_v, ssim_v, iou_v, dice_v, precision_v, recall_v, accuracy_v])

        # save output
        if args.save_comparison:
            if gt_bgr is not None:
                comp = np.concatenate((gt_bgr, lq_bgr, restored_bgr), axis=1)
            else:
                comp = np.concatenate((lq_bgr, restored_bgr), axis=1)
            out_path = os.path.join(args.output_dir, fname)
            cv2.imwrite(out_path, comp)
        else:
            out_path = os.path.join(args.output_dir, fname)
            cv2.imwrite(out_path, restored_bgr)

    # save csv
    df = pd.DataFrame(results, columns=['filename', 'psnr', 'ssim', 'iou', 'dice', 'precision', 'recall', 'accuracy'])
    csv_path = os.path.join(args.output_dir, 'restoration_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f'Done. Results saved to {args.output_dir}. CSV: {csv_path}')


if __name__ == '__main__':
    main()
