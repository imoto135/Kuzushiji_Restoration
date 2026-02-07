#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def crop_right_third(src_dir, dst_dir, exts):
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(src.iterdir()) if p.suffix.lower() in exts and p.is_file()]
    for p in tqdm(files, desc="Crop"):
        try:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            left = (w * 2) // 3
            box = (left, 0, w, h)
            cropped = im.crop(box)
            out_path = dst / p.name
            if out_path.suffix.lower() in ('.jpg', '.jpeg'):
                cropped.save(out_path, format='JPEG', quality=95, optimize=True)
            else:
                cropped.save(out_path)
        except Exception as e:
            print(f"Error processing {p.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Crop right 1/3 of images and save")
    parser.add_argument('--src', required=True, help='source directory (e.g. results/restored_net_g_140000)')
    parser.add_argument('--dst', required=False, help='destination dir (default: <src>/restored)', default=None)
    parser.add_argument('--exts', nargs='+', default=['.png', '.jpg', '.jpeg'], help='file extensions to process')
    args = parser.parse_args()

    dst = args.dst if args.dst else os.path.join(args.src, 'restored')
    exts = [e.lower() if e.startswith('.') else f'.{e.lower()}' for e in args.exts]
    crop_right_third(args.src, dst, exts)

if __name__ == '__main__':
    main()