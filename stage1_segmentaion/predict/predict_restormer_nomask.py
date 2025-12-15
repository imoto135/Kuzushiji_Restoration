#!/usr/bin/env python3
"""
Predict / restore images using a Restormer model trained without masks.

Saves:
  out_dir/restored/<stem>.png       -- restored RGB image (uint8)
  out_dir/comparisons/<stem>.png   -- side-by-side: original | restored

Example:
  python predict_restormer_nomask.py \
    --model-path restormer_nomask_best.pth \
    --data-dir dataset_final_hiragana \
    --lq-subdir lq_random/test \
    --image-size 128 \
    --out-dir results/restormer_nomask
"""
import os
import argparse
import logging
from PIL import Image
import numpy as np
import cv2
import torch
from tqdm import tqdm

from basicsr.models.archs.restormer_arch import Restormer


def build_map(d, allowed_exts={'.jpg', '.jpeg', '.png'}, pref_order=['.jpg', '.jpeg', '.png']):
    m = {}
    if not os.path.isdir(d):
        return m
    for fname in os.listdir(d):
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in allowed_exts:
            continue
        if stem not in m:
            m[stem] = fname
        else:
            cur_ext = os.path.splitext(m[stem])[1].lower()
            if pref_order.index(ext) < pref_order.index(cur_ext):
                m[stem] = fname
    return m


def load_state_dict_safe(model, path, strict=False):
    state = torch.load(path, map_location='cpu')
    sd = state.get('state_dict', state) if isinstance(state, dict) else state
    # strip 'module.' if present
    if isinstance(sd, dict):
        sd = { (k.replace('module.', '') if k.startswith('module.') else k): v for k, v in sd.items() }
        try:
            model.load_state_dict(sd, strict=strict)
            logging.info('Loaded state_dict into model (strict=%s)', strict)
            return True
        except Exception as e:
            logging.warning('Strict load failed: %s. Trying relaxed load with prefix stripping.', e)
            try:
                new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
                model.load_state_dict(new_sd, strict=False)
                logging.info('Loaded state_dict (relaxed strict=False)')
                return True
            except Exception as e2:
                logging.exception('Failed to load checkpoint: %s', e2)
                return False
    else:
        logging.warning('Checkpoint does not look like a state_dict mapping')
        return False


def prepare_tensor_from_pil(pil, image_size=None, device='cpu'):
    # pil: PIL Image RGB
    if image_size is not None and image_size > 0:
        pil = pil.resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(pil).astype(np.float32) / 255.0
    # HWC -> CHW
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return t


def main():
    parser = argparse.ArgumentParser(description='Predict / restore images using Restormer (no mask)')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='dataset_final_hiragana')
    parser.add_argument('--lq-subdir', type=str, default='lq_random/test')
    parser.add_argument('--image-size', type=int, default=128, help='if 0 use original size')
    parser.add_argument('--out-dir', type=str, default='results/restormer_nomask', help='output directory')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (auto if not set)')
    parser.add_argument('--use-amp', action='store_true', help='use mixed precision for faster inference')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    lq_dir = os.path.join(args.data_dir, args.lq_subdir)
    if not os.path.isdir(lq_dir):
        logging.error('Input directory not found: %s', lq_dir)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    out_restored = os.path.join(args.out_dir, 'restored')
    out_comp = os.path.join(args.out_dir, 'comparisons')
    os.makedirs(out_restored, exist_ok=True)
    os.makedirs(out_comp, exist_ok=True)

    img_map = build_map(lq_dir)
    stems = sorted(img_map.keys())
    if len(stems) == 0:
        logging.error('No input images found in %s', lq_dir)
        return

    # build model
    model = Restormer(inp_channels=3, out_channels=3)
    ok = load_state_dict_safe(model, args.model_path, strict=False)
    if not ok:
        logging.error('Failed to load checkpoint %s', args.model_path)
        return

    model = model.to(device)
    model.eval()

    use_amp = args.use_amp and device.type == 'cuda'

    for stem in tqdm(stems, desc='Predict'):
        try:
            fname = img_map[stem]
            path = os.path.join(lq_dir, fname)
            pil = Image.open(path).convert('RGB')
            orig_w, orig_h = pil.size

            inp = prepare_tensor_from_pil(pil, image_size=(args.image_size if args.image_size>0 else None), device=device)

            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = model(inp)[0]
                else:
                    out = model(inp)[0]
                out = torch.clamp(out, 0.0, 1.0)

            out_np = (out.squeeze(0).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
            # out_np is RGB

            # if resized earlier, resize back to original for comparison
            if args.image_size and args.image_size > 0:
                out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
                out_bgr = cv2.resize(out_bgr, (orig_w, orig_h))
            else:
                out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

            # save restored
            out_path = os.path.join(out_restored, f"{stem}.jpg")
            cv2.imwrite(out_path, out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # comparison: original | restored
            orig_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            # ensure same size
            if orig_bgr.shape[:2] != out_bgr.shape[:2]:
                out_bgr = cv2.resize(out_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]))

            comp = np.concatenate([orig_bgr, out_bgr], axis=1)
            comp_path = os.path.join(out_comp, f"{stem}.jpg")
            cv2.imwrite(comp_path, comp, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        except Exception as e:
            logging.exception('Failed to process %s: %s', stem, e)

    logging.info('Predictions saved to %s', args.out_dir)


if __name__ == '__main__':
    main()
