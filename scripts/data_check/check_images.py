#!/usr/bin/env python3
"""Scan images under a data directory for missing or corrupted files.

Usage: python3 scripts/data_check/check_images.py --root data
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def is_image_file(name: str) -> bool:
    return os.path.splitext(name.lower())[1] in EXTS


def find_images(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if is_image_file(fn):
                yield os.path.join(dirpath, fn)


def check_images(root: str):
    try:
        from PIL import Image
    except Exception as e:
        print('ERROR: Pillow is required but not installed:', e)
        print('Install with: pip install pillow')
        return 2

    total = 0
    zero_size = []
    verify_fail = []
    bad_dims = []
    counts_by_dir = defaultdict(int)

    for path in find_images(root):
        total += 1
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        rel = os.path.relpath(path, root)
        topdir = rel.split(os.sep)[0] if os.sep in rel else rel
        counts_by_dir[topdir] += 1

        if size == 0:
            zero_size.append(path)
            continue

        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as e:
            verify_fail.append((path, repr(e)))
            continue

        # reopen to get dimensions (verify() can close the file)
        try:
            with Image.open(path) as im:
                w, h = im.size
                if w == 0 or h == 0:
                    bad_dims.append(path)
        except Exception:
            verify_fail.append((path, 'open-failed-after-verify'))

    # Cross-check common paired folders if present
    pair_issues = []
    groups = ['gt', 'lq', 'gt_mask', 'pred_mask']
    subsets = ['train', 'val', 'test']
    for g in groups:
        base = os.path.join(root, g)
        if not os.path.isdir(base):
            continue
        for s in subsets:
            p = os.path.join(base, s)
            if not os.path.isdir(p):
                continue
            stems = set()
            for f in os.listdir(p):
                if is_image_file(f):
                    stems.add(os.path.splitext(f)[0])
            counts_by_dir[f'{g}/{s}'] = len(stems)

    for s in subsets:
        paths = {}
        for g in groups:
            p = os.path.join(root, g, s)
            if os.path.isdir(p):
                names = {os.path.splitext(f)[0] for f in os.listdir(p) if is_image_file(f)}
                paths[g] = names
        if 'gt' in paths and 'lq' in paths:
            only_gt = paths['gt'] - paths['lq']
            only_lq = paths['lq'] - paths['gt']
            if only_gt or only_lq:
                pair_issues.append((s, 'gt_vs_lq', len(paths['gt']), len(paths['lq']), len(only_gt), len(only_lq)))

    print('Image check report for:', root)
    print('  Total image files scanned:', total)
    if counts_by_dir:
        print('  Counts by top-level dir:')
        for k in sorted(counts_by_dir.keys()):
            print('   -', k, counts_by_dir[k])

    if zero_size:
        print('\nZero-size files (cannot be opened):')
        for p in zero_size[:50]:
            print(' -', p)
        if len(zero_size) > 50:
            print('  ... and', len(zero_size) - 50, 'more')

    if verify_fail:
        print('\nFiles failed verification/opening:')
        for p, e in verify_fail[:50]:
            print(' -', p, '=>', e)
        if len(verify_fail) > 50:
            print('  ... and', len(verify_fail) - 50, 'more')

    if bad_dims:
        print('\nFiles with zero width/height:')
        for p in bad_dims[:50]:
            print(' -', p)
        if len(bad_dims) > 50:
            print('  ... and', len(bad_dims) - 50, 'more')

    if pair_issues:
        print('\nPaired-folder mismatches (subset, type, gt_count, lq_count, only_gt, only_lq):')
        for it in pair_issues:
            print(' -', it)

    if total == 0:
        print('\nWARNING: No image files found under', root)

    problems = bool(zero_size or verify_fail or bad_dims or pair_issues)
    return 1 if problems else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data', help='root data directory to scan')
    args = p.parse_args()
    code = check_images(args.root)
    sys.exit(code)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Scan images under a data directory for missing or corrupted files.

Usage: python3 scripts/data_check/check_images.py --root data
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def is_image_file(name: str) -> bool:
    return os.path.splitext(name.lower())[1] in EXTS


def find_images(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if is_image_file(fn):
                yield os.path.join(dirpath, fn)


def check_images(root: str):
    try:
        from PIL import Image
    except Exception as e:
        print('ERROR: Pillow is required but not installed:', e)
        print('Install with: pip install pillow')
        return 2

    total = 0
    zero_size = []
    verify_fail = []
    bad_dims = []
    counts_by_dir = defaultdict(int)

    for path in find_images(root):
        total += 1
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        rel = os.path.relpath(path, root)
        topdir = rel.split(os.sep)[0] if os.sep in rel else rel
        counts_by_dir[topdir] += 1

        if size == 0:
            zero_size.append(path)
            continue

        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as e:
            verify_fail.append((path, repr(e)))
            continue

        # reopen to get dimensions (verify() can close the file)
        try:
            with Image.open(path) as im:
                w, h = im.size
                if w == 0 or h == 0:
                    bad_dims.append(path)
        except Exception:
            # already captured by verify in most cases, but keep guard
            verify_fail.append((path, 'open-failed-after-verify'))

    # Cross-check common paired folders if present
    pair_issues = []
    groups = ['gt', 'lq', 'gt_mask', 'pred_mask']
    subsets = ['train', 'val', 'test']
    for g in groups:
        base = os.path.join(root, g)
        if not os.path.isdir(base):
            continue
        for s in subsets:
            p = os.path.join(base, s)
            if not os.path.isdir(p):
                continue
            # collect stems
            stems = set()
            for f in os.listdir(p):
                if is_image_file(f):
                    stems.add(os.path.splitext(f)[0])
            # store for later comparison by group/subset
            counts_by_dir[f'{g}/{s}'] = len(stems)

    # For each subset, compare stems between gt and lq (if both exist)
    for s in subsets:
        paths = {}
        for g in groups:
            p = os.path.join(root, g, s)
            if os.path.isdir(p):
                names = {os.path.splitext(f)[0] for f in os.listdir(p) if is_image_file(f)}
                paths[g] = names
        if 'gt' in paths and 'lq' in paths:
            only_gt = paths['gt'] - paths['lq']
            only_lq = paths['lq'] - paths['gt']
            if only_gt or only_lq:
                pair_issues.append((s, 'gt_vs_lq', len(paths['gt']), len(paths['lq']), len(only_gt), len(only_lq)))

    # Print report
    print('Image check report for:', root)
    print('  Total image files scanned:', total)
    if counts_by_dir:
        print('  Counts by top-level dir:')
        for k in sorted(counts_by_dir.keys()):
            print('   -', k, counts_by_dir[k])

    if zero_size:
        print('\nZero-size files (cannot be opened):')
        for p in zero_size[:50]:
            print(' -', p)
        if len(zero_size) > 50:
            print('  ... and', len(zero_size) - 50, 'more')

    if verify_fail:
        print('\nFiles failed verification/opening:')
        for p, e in verify_fail[:50]:
            print(' -', p, '=>', e)
        if len(verify_fail) > 50:
            print('  ... and', len(verify_fail) - 50, 'more')

    if bad_dims:
        print('\nFiles with zero width/height:')
        for p in bad_dims[:50]:
            print(' -', p)
        if len(bad_dims) > 50:
            print('  ... and', len(bad_dims) - 50, 'more')

    if pair_issues:
        print('\nPaired-folder mismatches (subset, type, gt_count, lq_count, only_gt, only_lq):')
        for it in pair_issues:
            print(' -', it)

    if total == 0:
        print('\nWARNING: No image files found under', root)

    # exit code: 0 if no problems, 1 if any problems, 2 if missing dependency
    problems = bool(zero_size or verify_fail or bad_dims or pair_issues)
    return 1 if problems else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data', help='root data directory to scan')
    args = p.parse_args()
    code = check_images(args.root)
    sys.exit(code)


if __name__ == '__main__':
    main()
