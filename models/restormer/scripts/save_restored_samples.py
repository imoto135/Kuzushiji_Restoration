#!/usr/bin/env python3
"""
Save restored images only using the same restoration procedure as
`scripts/apply_restormer_with_masks.py`.

Saves outputs to `results/samples/` by default. Can limit number of samples
with --max-samples.
"""
import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2

# import Restormer architecture from the project
from basicsr.models.archs.restormer_arch import Restormer


def find_latest_checkpoint(models_dir):
    if not os.path.isdir(models_dir):
        return None
    files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not files:
        return None
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
    state = None
    for k in ('params', 'state_dict', 'model', 'net_g'):
        if isinstance(ck, dict) and k in ck:
            state = ck[k]
            break
    if state is None and isinstance(ck, dict):
        state = ck
    if state is None:
        raise RuntimeError(f'Unable to find model state in checkpoint: {path}')

    new_state = {}
    for k, v in state.items():
        new_k = k[len('module.'):] if k.startswith('module.') else k
        new_state[new_k] = v

    model.load_state_dict(new_state)
    model.to(device)
    model.eval()


def img_to_tensor_rgb(img_pil):
    tensor = transforms.ToTensor()(img_pil)
    return tensor


def img_to_tensor_mask(mask_pil):
    mask = np.array(mask_pil.convert('L'), dtype=np.float32) / 255.0
    mask = np.expand_dims(mask, 0)
    return torch.from_numpy(mask)


def find_mask_file_for(fn, mask_dir):
    stem = os.path.splitext(fn)[0]
    exts = ('.jpg', '.png', '.jpeg')

    cand = os.path.join(mask_dir, fn)
    if os.path.isfile(cand):
        return cand

    suffixes = ['', '_prediction', '_mask']
    for suf in suffixes:
        for ext in exts:
            cand = os.path.join(mask_dir, stem + suf + ext)
            if os.path.isfile(cand):
                return cand

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

    try:
        for root, dirs, files in os.walk(mask_dir):
            for f in files:
                name_stem = os.path.splitext(f)[0]
                if name_stem == stem or name_stem.startswith(stem) or stem.startswith(name_stem):
                    return os.path.join(root, f)
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to restormer checkpoint (.pth).')
    parser.add_argument('--models-dir', type=str, default='experiments/UNet_Restormer_Hiragana_Run01/models')
    parser.add_argument('--input-lq', type=str, default='dataset_final_hiragana/lq_random/test')
    parser.add_argument('--input-gt', type=str, default='dataset_final_hiragana/gt/test', help='GT images directory; used to name outputs')
    parser.add_argument('--mask-root', type=str, default='dataset_final_hiragana/mask_random_prediction')
    parser.add_argument('--mask-subset', type=str, default='test')
    parser.add_argument('--output-dir', type=str, default='results/samples')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--max-samples', type=int, default=0, help='Max number of samples to process (0 = all)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    if args.checkpoint is None:
        ck = find_latest_checkpoint(args.models_dir)
        if ck is None:
            raise SystemExit('No checkpoint found; provide --checkpoint or ensure models dir exists')
        checkpoint_path = ck
    else:
        checkpoint_path = args.checkpoint

    os.makedirs(args.output_dir, exist_ok=True)

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

    max_samples = args.max_samples if args.max_samples > 0 else len(files)

    # build gt mapping for quick name lookup
    gt_files = {}
    if input_gt_dir:
        import glob
        for p in glob.glob(os.path.join(input_gt_dir, '*')):
            if os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg'):
                gt_files[os.path.splitext(os.path.basename(p))[0]] = p

    def find_matching_gt(restored_fname):
        """Try to find GT filename that matches the given LQ/restored filename by stem.
        Returns the GT basename (filename only) or None.
        """
        stem = os.path.splitext(restored_fname)[0]
        # exact
        if stem in gt_files:
            return os.path.basename(gt_files[stem])
        # try common suffix removals
        for suf in ['_prediction', '_pred', '_restored', '_recon', '_out', '_lq']:
            if stem.endswith(suf):
                key = stem[:-len(suf)]
                if key in gt_files:
                    return os.path.basename(gt_files[key])
        # substring match
        for k, v in gt_files.items():
            if k in stem or stem in k:
                return os.path.basename(v)
        return None

    to_tensor = transforms.ToTensor()
    cnt = 0
    for fname in tqdm(files, desc='Restoring'):
        if cnt >= max_samples:
            break
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

        # prefer to save using the GT filename (same basename and extension) when available
        out_name = None
        if input_gt_dir:
            matched_gt_basename = find_matching_gt(fname)
            if matched_gt_basename:
                out_name = matched_gt_basename
        if out_name is None:
            out_name = fname
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, restored_bgr)
        cnt += 1

    print(f'Done. Saved {cnt} restored images to {args.output_dir}')


if __name__ == '__main__':
    main()
