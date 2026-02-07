import os
import re
import argparse
from typing import Dict, Set, List, Tuple

import yaml


DEFAULT_SUFFIXES = [
    "_Missing",
    "_restored",
    "_prediction",
    "_pred",
    "_out",
]


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def normalize_base(s: str, suffixes: List[str]) -> str:
    """拡張子除去後、末尾のサフィックスを削って正規化した base を返す。"""
    base = stem(s)
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if base.endswith(suf):
                base = base[: -len(suf)]
                changed = True
    return base


def list_bases(dir_path: str, suffixes: List[str]) -> Tuple[Set[str], Dict[str, str]]:
    """
    dir内のファイルから正規化base集合と、base->代表ファイル名の対応を作る
    （重複baseがある場合は最初に見つかったものを代表にする）
    """
    bases: Set[str] = set()
    rep: Dict[str, str] = {}
    for f in os.listdir(dir_path):
        if f.startswith("."):
            continue
        p = os.path.join(dir_path, f)
        if not os.path.isfile(p):
            continue
        b = normalize_base(f, suffixes)
        bases.add(b)
        rep.setdefault(b, f)
    return bases, rep


def generate_for_phase(phase: str, opt: dict, suffixes: List[str]) -> None:
    gt_dir = opt["dataroot_gt"]
    lq_dir = opt["dataroot_lq"]
    mask_dir = opt["dataroot_mask"]
    out = opt["filelist_path"]

    print(f"\n[phase={phase}]")
    print(f"  GT  : {gt_dir}")
    print(f"  LQ  : {lq_dir}")
    print(f"  MASK: {mask_dir}")
    print(f"  OUT : {out}")

    for name, d in [("GT", gt_dir), ("LQ", lq_dir), ("MASK", mask_dir)]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{name} dir missing: {d}")

    gt_bases, _ = list_bases(gt_dir, suffixes)
    lq_bases, _ = list_bases(lq_dir, suffixes)
    mask_bases, _ = list_bases(mask_dir, suffixes)

    print(f"  counts: gt={len(gt_bases)} lq={len(lq_bases)} mask={len(mask_bases)}")

    selected = sorted(gt_bases & lq_bases & mask_bases)
    print(f"  selected count: {len(selected)}")

    # backup existing
    if os.path.exists(out):
        bak = out + ".bak"
        os.replace(out, bak)
        print(f"  backed up: {out} -> {bak}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in selected:
            f.write(b + "\n")
    print(f"  wrote: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="restormer/configs/restormer_config.yml",
        help="Restormer dataset config yml",
    )
    ap.add_argument(
        "--suffix",
        action="append",
        default=[],
        help="suffix to strip (can be specified multiple times)",
    )
    args = ap.parse_args()

    suffixes = DEFAULT_SUFFIXES + args.suffix

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    for phase in ("train", "val"):
        opt = cfg["datasets"][phase]
        generate_for_phase(phase, opt, suffixes)

    print("\ndone")


if __name__ == "__main__":
    main()
