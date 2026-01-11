
# filepath: /home/ihpc/imoto/Kuzushiji_Restoration/stage2_restoration/infer/infer_restormer_withmask.py
#!/usr/bin/env python3
import os
import argparse
import logging
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from restormer.basicsr.models.archs.restormer_arch import Restormer


def build_file_map(d, allowed_exts={'.jpg', '.jpeg', '.png'}, pref_order=('.jpg', '.jpeg', '.png')):
    m = {}
    if not os.path.isdir(d):
        return m

    def _normalize_stem(stem):
        return re.sub(r'([_\-][A-Za-z]+)$', '', stem)

    for fname in os.listdir(d):
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in allowed_exts:
            continue
        key = _normalize_stem(stem)
        if key not in m:
            m[key] = fname
        else:
            cur_ext = os.path.splitext(m[key])[1].lower()
            try:
                if pref_order.index(ext) < pref_order.index(cur_ext):
                    m[key] = fname
            except ValueError:
                pass
    return m


class LQMaskDataset(Dataset):
    """推論用: lq (RGB) + mask (L) -> 入力テンソル, 出力ファイル名."""
    def __init__(self, lq_dir, mask_dir, image_size=None):
        self.lq_dir = lq_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        if not os.path.isdir(lq_dir) or not os.path.isdir(mask_dir):
            logging.error(f"ディレクトリが見つかりません。 LQ: {lq_dir}, MASK: {mask_dir}")
            self.pairs = []
            return

        lq_map = build_file_map(lq_dir)
        mask_map = build_file_map(mask_dir)
        common = sorted(set(lq_map.keys()).intersection(mask_map.keys()))
        self.pairs = [(lq_map[k], mask_map[k], k) for k in common]
        if len(self.pairs) == 0:
            logging.error(f"共通するファイルが見つかりません: {lq_dir}, {mask_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lq_fname, mask_fname, key = self.pairs[idx]
        lq_path = os.path.join(self.lq_dir, lq_fname)
        mask_path = os.path.join(self.mask_dir, mask_fname)

        lq_img = Image.open(lq_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')

        if self.image_size is not None and self.image_size > 0:
            lq_img = lq_img.resize((self.image_size, self.image_size), Image.BICUBIC)
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)

        lq = np.array(lq_img, dtype=np.float32) / 255.0
        mask = np.array(mask_img, dtype=np.uint8)
        # 学習と同じく 2値化（>127 → 1）
        mask = (mask > 127).astype(np.float32)

        lq = torch.from_numpy(lq.transpose(2, 0, 1)).float()          # (3, H, W)
        mask = torch.from_numpy(mask[None, ...]).float()              # (1, H, W)
        inp = torch.cat([lq, mask], dim=0)                             # (4, H, W)

        # 保存時に使う出力ファイル名（拡張子は .png に統一）
        out_name = f"{key}.png"
        return inp, out_name


def load_model(weights_path, device):
    model = Restormer(inp_channels=4, out_channels=3)
    state = torch.load(weights_path, map_location=device)
    sd = state.get('state_dict', state) if isinstance(state, dict) else state
    if isinstance(sd, dict):
        sd = { (k.replace('module.', '') if k.startswith('module.') else k): v
               for k, v in sd.items() }
        model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference(args):
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logging.info(f'Using device: {device}')

    lq_dir = os.path.join(args.data_dir, args.lq_dir)
    mask_dir = os.path.join(args.data_dir, args.mask_dir)

    dataset = LQMaskDataset(lq_dir, mask_dir, image_size=args.image_size)
    if len(dataset) == 0:
        logging.error('推論用データセットが空です。パスを確認してください。')
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = load_model(args.weights, device)
    logging.info(f'Loaded weights from {args.weights}')

    use_amp = args.use_amp and (device.type == 'cuda')

    with torch.no_grad():
        for inps, out_names in tqdm(loader, desc='Inference'):
            inps = inps.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outs = model(inps)[0]
            else:
                outs = model(inps)[0]

            outs = torch.clamp(outs, 0.0, 1.0).cpu()

            for img_tensor, name in zip(outs, out_names):
                img = img_tensor.permute(1, 2, 0).numpy()   # HWC
                img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(args.out_dir, name))

    logging.info('Inference finished.')


def parse_args():
    p = argparse.ArgumentParser(description='Restormer mask-guided inference')
    p.add_argument('--weights', type=str, required=True,
                   help='学習済み重み (.pth)')
    p.add_argument('--data-dir', type=str, default='hiragana_dataset')
    p.add_argument('--lq-dir', type=str, default='lq/val',
                   help='data-dir からの相対パス (LQ 画像)')
    p.add_argument('--mask-dir', type=str, default='gt_mask/val',
                   help='data-dir からの相対パス (マスク画像)')
    p.add_argument('--out-dir', type=str, default='outputs/restormer_withmask',
                   help='復元画像の出力先ディレクトリ')
    p.add_argument('--image-size', type=int, default=128,
                   help='入力画像をこのサイズにリサイズ（0以下でリサイズなし）')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--use-amp', action='store_true',
                   help='CUDA 使用時に自動混合精度を有効化')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(args)