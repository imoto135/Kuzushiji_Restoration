"""
joint_dataset.py — KuzushijiJointDataset

Returns (lq, gt, gt_mask) triplets for end-to-end joint training.

Directory layout expected (identical to the existing project structure):
    data/hiragana_fulldataset_5stain/
        lq/{train,val,test}/    ← degraded RGB images  [0,255]
        gt/{train,val,test}/    ← clean RGB images      [0,255]
        gt_mask/{train,val,test}/  ← binary damage masks [0,255]

All images are returned as float32 tensors in [0, 1].
"""

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(img_size: int = 128, mode: str = "train") -> A.Compose:
    """
    Augmentation pipeline shared across lq / gt / mask images.

    - No horizontal flip (mirror images look like different characters)
    - Small rotation (±10°) to simulate natural writing tilt
    - No ImageNet normalisation here — lq is kept in [0,1] for NAFNet;
      UNet++ normalisation is applied inside JointRestorationNet.forward()
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Rotate(limit=10, p=0.3),
            ],
            additional_targets={"gt": "image", "mask": "mask"},
        )
    else:
        return A.Compose(
            [A.Resize(img_size, img_size)],
            additional_targets={"gt": "image", "mask": "mask"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KuzushijiJointDataset(Dataset):
    """
    Triplet dataset: (lq, gt, gt_mask).

    File matching logic is the same as in models/unet++/train.py:
    strip noise-type prefix and damage-type suffix from the filename stem
    before matching across directories.
    """

    _PREFIX_RE = re.compile(r"^type\d+_sev[\d\.]+_")
    _SUFFIX_RE = re.compile(
        r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain|Abrasion|Transparent)$"
    )

    def __init__(
        self,
        lq_dir: Union[str, Path],
        gt_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        transform: Optional[A.Compose] = None,
        img_size: int = 128,
        mode: str = "train",
    ):
        self.lq_dir   = Path(lq_dir)
        self.gt_dir   = Path(gt_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform or get_transforms(img_size, mode)
        self.triplets  = self._build_triplets()

    # ------------------------------------------------------------------
    # File matching
    # ------------------------------------------------------------------

    @classmethod
    def _clean_stem(cls, path: Path) -> str:
        stem = path.stem
        stem = cls._PREFIX_RE.sub("", stem)
        stem = cls._SUFFIX_RE.sub("",  stem)
        return stem

    def _index(self, directory: Path) -> dict[str, Path]:
        return {
            self._clean_stem(f): f
            for f in directory.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

    def _build_triplets(self) -> list[tuple[Path, Path, Path]]:
        lq_idx   = self._index(self.lq_dir)
        gt_idx   = self._index(self.gt_dir)
        mask_idx = self._index(self.mask_dir)

        common = set(lq_idx) & set(gt_idx) & set(mask_idx)
        if not common:
            raise FileNotFoundError(
                f"No matching triplets found.\n"
                f"  lq:   {self.lq_dir}\n"
                f"  gt:   {self.gt_dir}\n"
                f"  mask: {self.mask_dir}"
            )

        triplets = sorted(
            [(lq_idx[k], gt_idx[k], mask_idx[k]) for k in common],
            key=lambda t: t[0].name,
        )
        print(f"[JointDataset] Found {len(triplets)} triplets in {self.lq_dir.parent.name}/{self.lq_dir.name}")
        return triplets

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        lq_path, gt_path, mask_path = self.triplets[idx]

        lq   = np.array(Image.open(lq_path).convert("RGB"),  dtype=np.uint8)
        gt   = np.array(Image.open(gt_path).convert("RGB"),  dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"),   dtype=np.float32) / 255.0

        if self.transform:
            aug    = self.transform(image=lq, gt=gt, mask=mask)
            lq     = aug["image"]
            gt     = aug["gt"]
            mask   = aug["mask"]

        # Convert to float32 [0, 1] tensors — contiguous to avoid RuntimeError
        def to_tensor_chw(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(
                np.ascontiguousarray(arr).transpose(2, 0, 1), dtype=torch.float32
            ) / 255.0

        def to_tensor_hw(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(
                np.ascontiguousarray(arr), dtype=torch.float32
            ).unsqueeze(0)

        return {
            "lq":      to_tensor_chw(lq),    # (3, H, W) in [0,1]
            "gt":      to_tensor_chw(gt),    # (3, H, W) in [0,1]
            "gt_mask": to_tensor_hw(mask),   # (1, H, W) in [0,1]
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_base = Path(__file__).resolve().parents[2] / "data" / "hiragana_fulldataset_5stain"
    ds = KuzushijiJointDataset(
        lq_dir   = data_base / "lq"      / "val",
        gt_dir   = data_base / "gt"      / "val",
        mask_dir = data_base / "gt_mask" / "val",
        mode     = "val",
    )
    print(f"Dataset len: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: {v.shape}  range [{v.min():.3f}, {v.max():.3f}]")
    print("Sanity check passed ✓")
