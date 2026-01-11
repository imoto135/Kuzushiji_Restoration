#!/usr/bin/env python3
"""
convert_png_to_jpg.py

Usage:
  python3 scripts/convert_png_to_jpg.py <root_dir> [--workers N] [--quality Q] [--limit N] [--skip-existing]

This script walks <root_dir> recursively, finds PNG files and converts them to JPEG.
It preserves directory structure and creates .jpg files next to the pngs. The pngs are
kept by default.

Alpha channel is composited over white.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image


def convert_one(path, quality=95, skip_existing=True, remove_original=False):
    jpg_path = os.path.splitext(path)[0] + '.jpg'
    if skip_existing and os.path.exists(jpg_path):
        if remove_original and os.path.exists(path):
            try:
                os.remove(path)
                return ('removed', path)
            except Exception as e:
                return ('error', path, f'delete failed: {e}')
        return ('skipped', path)
    try:
        with Image.open(path) as im:
            # Convert paletted or others to RGBA if needed
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                bg = Image.new('RGB', im.size, (255, 255, 255))
                rgba = im.convert('RGBA')
                bg.paste(rgba, mask=rgba.split()[3])
                out = bg
            else:
                out = im.convert('RGB')

            tmp = jpg_path + '.tmp'
            out.save(tmp, format='JPEG', quality=quality)
            os.replace(tmp, jpg_path)
        if remove_original:
            try:
                os.remove(path)
            except Exception as e:
                return ('error', path, f'delete failed: {e}')
        return ('ok', path)
    except Exception as e:
        return ('error', path, str(e))


def find_pngs(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.png'):
                yield os.path.join(dirpath, fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--quality', type=int, default=95)
    p.add_argument('--limit', type=int, default=0, help='If >0, only convert this many files (useful for testing)')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--remove-original', action='store_true', help='Remove original PNG after successful conversion')
    args = p.parse_args()

    pngs = list(find_pngs(args.root))
    total = len(pngs)
    print(f'Found {total} PNG files under {args.root!r}')

    if args.limit and args.limit > 0:
        pngs = pngs[:args.limit]
        print(f'Limiting to first {len(pngs)} files for this run')

    if args.dry_run:
        for pth in pngs[:50]:
            print(pth)
        print('Dry run complete')
        return

    ok = 0
    skipped = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_one, path, args.quality, args.skip_existing, args.remove_original): path for path in pngs}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res[0] in ('ok', 'removed'):
                ok += 1
            elif res[0] == 'skipped':
                skipped += 1
            else:
                errors.append(res)

            if i % 100 == 0 or i == len(pngs):
                print(f'Processed {i}/{len(pngs)} (ok={ok} skipped={skipped} errors={len(errors)})')

    print('Done')
    print(f'ok={ok} skipped={skipped} errors={len(errors)}')
    if errors:
        print('Some errors:')
        for e in errors[:20]:
            print(e)



if __name__ == '__main__':
    main()
