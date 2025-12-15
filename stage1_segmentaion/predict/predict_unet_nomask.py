import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

# --- simple UNet ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNetMini(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ConvBlock(base_ch*4 + base_ch*2, base_ch*2)
        self.dec2 = ConvBlock(base_ch*2 + base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.up(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.outc(d2)
        return torch.sigmoid(out)

# --- utils ---
def list_images(d, exts=('.png','.jpg','.jpeg')):
    files = []
    for e in exts:
        files += glob(os.path.join(d, f'*{e}'))
    files = sorted(files)
    return files

def load_checkpoint(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location='cpu')
    sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
    # remove module. prefix
    sd = { (k.replace('module.', '') if k.startswith('module.') else k): v for k,v in sd.items() }
    model_sd = model.state_dict()
    # keep only matching keys
    new_sd = {}
    for k,v in sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            new_sd[k]=v
    model.load_state_dict(new_sd, strict=False)
    model.to(device)

# --- main ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, default=None, help='(.pth) optional pretrained weights')
    p.add_argument('--data-dir', type=str, required=True, help='dataset root containing images')
    p.add_argument('--lq-subdir', type=str, default='lq/test', help='input images subdir under data-dir')
    p.add_argument('--out-dir', type=str, default='results/unet_nomask', help='output dir')
    p.add_argument('--image-size', type=int, default=256, help='resize short side to this and center crop to square')
    p.add_argument('--threshold', type=float, default=0.5, help='binarization threshold')
    p.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    in_dir = os.path.join(args.data_dir, args.lq_subdir)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'vis'), exist_ok=True)

    img_paths = list_images(in_dir)
    if len(img_paths) == 0:
        print('No images found in', in_dir); return

    # model
    model = UNetMini(in_ch=3, base_ch=32)
    if args.model_path:
        load_checkpoint(model, args.model_path, device)
    model.to(device).eval()

    tf = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
    ])

    for p in tqdm(img_paths, desc='Predict'):
        name = os.path.splitext(os.path.basename(p))[0]
        img = Image.open(p).convert('RGB')
        inp = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)  # [1,1,H,W]
        out_np = out.squeeze().cpu().numpy()
        bin_mask = (out_np >= args.threshold).astype(np.uint8) * 255
        mask_p = os.path.join(args.out_dir, 'masks', f'{name}.png')
        Image.fromarray(bin_mask).convert('L').save(mask_p)

        # visualization side-by-side (resize original to image-size)
        vis_img = np.array(img.resize((args.image_size, args.image_size)))
        mask_rgb = np.stack([bin_mask]*3, axis=2)
        overlay = (vis_img * 0.6 + mask_rgb * 0.4).astype(np.uint8)
        vis = np.concatenate([vis_img, overlay], axis=1)
        Image.fromarray(vis).save(os.path.join(args.out_dir, 'vis', f'{name}.png'))

    print('Saved masks ->', os.path.join(args.out_dir, 'masks'))

if __name__ == '__main__':
    main()