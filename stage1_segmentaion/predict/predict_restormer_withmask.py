#!/usr/bin/env python3
"""
Restormer (mask-guided) 推論スクリプト

使い方例:
python predict_restormer_withmask.py \
  --model-path restormer_withmask_best.pth \
  --data-dir dataset_final_hiragana \
  --lq-subdir lq_random/test --mask-subdir mask_gt/test \
  --image-size 128 --batch-size 4 --out-dir results/restormer_withmask \
  --use-amp
"""
import os
import sys
import argparse
import logging
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# allow importing local basicsr if present
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from basicsr.models.archs.restormer_arch import Restormer


def normalize_stem(stem):
    # remove common prediction/mask suffixes so stems match across dirs
    for s in ('_prediction', '_pred', '_mask', '_mask_gt'):
        if stem.endswith(s):
            return stem[: -len(s)]
    return stem


def build_file_map(d, allowed_exts={'.jpg', '.jpeg', '.png'}, pref_order=['.jpg', '.jpeg', '.png']):
    m = {}
    if not os.path.isdir(d):
        return m
    for fname in os.listdir(d):
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in allowed_exts:
            continue
        norm = normalize_stem(stem)
        # prefer ext order if multiple candidates map to same normalized stem
        if norm not in m:
            m[norm] = fname
        else:
            cur_ext = os.path.splitext(m[norm])[1].lower()
            try:
                if pref_order.index(ext) < pref_order.index(cur_ext):
                    m[norm] = fname
            except ValueError:
                m[norm] = fname
    return m


class InferenceMaskDataset(Dataset):
    def __init__(self, lq_dir, mask_dir, image_size=0):
        self.lq_dir = lq_dir
        self.mask_dir = mask_dir
        self.image_size = int(image_size)
        lq_map = build_file_map(lq_dir)
        mask_map = build_file_map(mask_dir)
        self.common = sorted(set(lq_map.keys()).intersection(set(mask_map.keys())))
        self.lq_map = lq_map
        self.mask_map = mask_map

    def __len__(self):
        return len(self.common)

    def __getitem__(self, idx):
        stem = self.common[idx]
        lq_path = os.path.join(self.lq_dir, self.lq_map[stem])
        mask_path = os.path.join(self.mask_dir, self.mask_map[stem])

        lq_img = Image.open(lq_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')

        if self.image_size and (lq_img.size[0] != self.image_size or lq_img.size[1] != self.image_size):
            lq_img = lq_img.resize((self.image_size, self.image_size), Image.BICUBIC)
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)

        lq = np.array(lq_img, dtype=np.float32) / 255.0  # HWC
        mask = np.array(mask_img, dtype=np.uint8)

        # return originals for saving + normalized tensors
        return stem, lq, mask


def load_state_safe(model, path, strict=False):
    try:
        state = torch.load(path, map_location='cpu')
        sd = state.get('state_dict', state) if isinstance(state, dict) else state
        if isinstance(sd, dict):
            sd = { (k.replace('module.', '') if k.startswith('module.') else k): v for k, v in sd.items() }
            model.load_state_dict(sd, strict=strict)
            logging.info(f'Loaded pretrained weights from {path} (strict={strict})')
            return True
    except Exception as e:
        logging.warning(f'Failed to load pretrained weights: {e}')
    return False


def make_dirs(out_dir):
    restored_dir = os.path.join(out_dir, 'restored')
    comps_dir = os.path.join(out_dir, 'comparisons')
    os.makedirs(restored_dir, exist_ok=True)
    os.makedirs(comps_dir, exist_ok=True)
    return restored_dir, comps_dir


def to_uint8(img):
    im = np.clip((img * 255.0).round(), 0, 255).astype(np.uint8)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='experiments/UNet_Restormer_Hiragana_Run01/models/net_g_140000.pth')
    parser.add_argument('--data-dir', type=str, default='dataset_final_hiragana')
    parser.add_argument('--lq-subdir', type=str, default='/lq_random/test')
    parser.add_argument('--mask-subdir', type=str, default='/mask_random_prediction/test')
    parser.add_argument('--image-size', type=int, default=128, help='0 = keep original')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-dir', type=str, default='results/restormer_withpredmask_140000')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--mask-thresh', type=float, default=0.5, help='threshold for mask binarization (0..1)')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    logging.info(f'Using device: {device}')

    lq_dir = os.path.join(args.data_dir, args.lq_subdir)
    mask_dir = os.path.join(args.data_dir, args.mask_subdir)

    ds = InferenceMaskDataset(lq_dir, mask_dir, image_size=args.image_size)
    if len(ds) == 0:
        logging.error('No common files found between LQ and mask dirs.')
        return

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # build model
    model = Restormer(inp_channels=4, out_channels=3)
    model = model.to(device)
    load_state_safe(model, args.model_path, strict=False)
    model.eval()

    restored_dir, comps_dir = make_dirs(args.out_dir)
    mask_thresh_255 = int(args.mask_thresh * 255.0)

    use_amp = args.use_amp and device.type == 'cuda'
    with torch.no_grad():
        for batch in tqdm(dl, desc='Infer'):
            stems, lqs, masks = batch
            batch_tensors = []
            origs = []
            masks_vis = []
            for lq, m in zip(lqs, masks):
                # --- normalize lq to numpy HWC float32 in 0..1 ---
                if isinstance(lq, torch.Tensor):
                    lq_np = lq.cpu().numpy()
                    # if CHW -> HWC
                    if lq_np.ndim == 3 and lq_np.shape[0] in (1, 3, 4):
                        lq_np = lq_np.transpose(1, 2, 0)
                else:
                    lq_np = np.asarray(lq)
                lq_np = lq_np.astype(np.float32)
                if lq_np.max() > 1.1:
                    lq_np = lq_np / 255.0
                # ensure HWC with 3 channels
                if lq_np.ndim == 2:
                    lq_np = np.stack([lq_np] * 3, axis=2)
                if lq_np.shape[2] == 4:
                    lq_np = lq_np[..., :3]

                # --- normalize mask to binary numpy (0/1) ---
                if isinstance(m, torch.Tensor):
                    m_np = m.cpu().numpy()
                else:
                    m_np = np.asarray(m)
                if m_np.max() > 1:
                    mb = (m_np > mask_thresh_255).astype(np.float32)
                else:
                    mb = (m_np > 0).astype(np.float32)

                # build input tensor CHW, concat mask as 1st/last channel
                chw = torch.from_numpy(lq_np.transpose(2, 0, 1)).float()
                mb_t = torch.from_numpy(mb[None, ...]).float()
                inp = torch.cat([chw, mb_t], dim=0)
                batch_tensors.append(inp)
                origs.append(lq_np)
                masks_vis.append(mb)
                inp_batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(inp_batch)
            else:
                out = model(inp_batch)
            # model may return tuple/list
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = torch.clamp(out, 0.0, 1.0).cpu().numpy()  # N x C x H x W

            for i, stem in enumerate(stems):
                pred_chw = out[i]
                pred_hwc = pred_chw.transpose(1, 2, 0)  # HWC float
                pred_u8 = to_uint8(pred_hwc)
                orig_u8 = to_uint8(origs[i])
                mask_u8 = (masks_vis[i] * 255).astype(np.uint8)

                # save restored
                restored_path = os.path.join(restored_dir, f'{stem}.jpg')
                Image.fromarray(pred_u8).convert('RGB').save(restored_path, format='JPEG', quality=95)

                # build comparison (orig | mask_vis | restored)
                mask_rgb = np.stack([mask_u8]*3, axis=2)
                comp = np.concatenate([orig_u8, mask_rgb, pred_u8], axis=1)
                comp_path = os.path.join(comps_dir, f'{stem}.jpg')
                Image.fromarray(comp).convert('RGB').save(comp_path, format='JPEG', quality=95)

    logging.info(f'Done. Restored saved to: {restored_dir}, comparisons: {comps_dir}')


if __name__ == '__main__':
    main()