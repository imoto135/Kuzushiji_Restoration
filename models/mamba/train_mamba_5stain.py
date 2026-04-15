"""
train_mamba_5stain.py — MambaIR on hiragana_fulldataset_5stain

3パターン対応:
  --mask_type nomask   : 3ch input (RGB only)
  --mask_type predmask : 4ch input (RGB + pred_mask)
  --mask_type gtmask   : 4ch input (RGB + gt_mask)

他モデルと設定を揃えた:
  - AdamW lr=2e-4, betas=(0.9, 0.9), weight_decay=0
  - CosineAnnealingLR, total_iters=200000, eta_min=1e-7
  - Loss: Charbonnier + 0.1 * LPIPS(vgg)
  - batch_size=16, img_size=128
  - bfloat16 autocast
"""

import argparse
import math
import os
from pathlib import Path
import re

import albumentations as A
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mambair import MambaIR


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class MambaDataset(Dataset):
    _PREFIX_RE = re.compile(r"^type\d+_sev[\d\.]+_")
    _SUFFIX_RE = re.compile(
        r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain|Abrasion|Transparent)$"
    )

    def __init__(self, lq_dir, gt_dir, mask_dir=None, img_size=128, mode="train"):
        """
        mask_dir=None の場合は NoMask (3ch入力)
        mask_dir が指定された場合は 4ch入力 (RGB + mask)
        """
        self.lq_dir = Path(lq_dir)
        self.gt_dir = Path(gt_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None

        if mode == "train":
            self.transform = A.Compose(
                [A.Resize(img_size, img_size), A.Rotate(limit=10, p=0.3)],
                additional_targets={"gt": "image", "mask": "mask"},
            )
        else:
            self.transform = A.Compose(
                [A.Resize(img_size, img_size)],
                additional_targets={"gt": "image", "mask": "mask"},
            )

        self.pairs = self._build_pairs()

    @classmethod
    def _clean_stem(cls, path: Path) -> str:
        stem = path.stem
        stem = cls._PREFIX_RE.sub("", stem)
        stem = cls._SUFFIX_RE.sub("", stem)
        return stem

    def _index(self, directory: Path) -> dict:
        return {
            self._clean_stem(f): f
            for f in directory.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

    def _build_pairs(self):
        lq_idx = self._index(self.lq_dir)
        gt_idx = self._index(self.gt_dir)
        common = set(lq_idx) & set(gt_idx)

        if self.mask_dir is not None:
            mask_idx = self._index(self.mask_dir)
            common = common & set(mask_idx)
            pairs = sorted(
                [(lq_idx[k], gt_idx[k], mask_idx[k]) for k in common]
            )
        else:
            pairs = sorted([(lq_idx[k], gt_idx[k], None) for k in common])

        print(f"[Dataset] {self.lq_dir.parent.name}/{self.lq_dir.name}: {len(pairs)} pairs")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lq_p, gt_p, mask_p = self.pairs[idx]

        lq_img = np.array(Image.open(lq_p).convert("RGB"))
        gt_img = np.array(Image.open(gt_p).convert("RGB"))

        if mask_p is not None:
            mask_m = np.array(Image.open(mask_p).convert("L"))
            aug = self.transform(image=lq_img, gt=gt_img, mask=mask_m)
            mask_t = torch.from_numpy(aug["mask"]).unsqueeze(0).float() / 255.0
        else:
            aug = self.transform(image=lq_img, gt=gt_img, mask=np.zeros(lq_img.shape[:2], dtype=np.float32))
            mask_t = None

        lq_t = torch.from_numpy(aug["image"]).permute(2, 0, 1).float() / 255.0
        gt_t = torch.from_numpy(aug["gt"]).permute(2, 0, 1).float() / 255.0

        if mask_t is not None:
            input_t = torch.cat([lq_t, mask_t], dim=0)  # [4, H, W]
        else:
            input_t = lq_t  # [3, H, W]

        return {"input": input_t, "gt": gt_t}


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# -----------------------------------------------------------------------------
# PSNR
# -----------------------------------------------------------------------------
def calculate_psnr(pred, gt):
    pred = pred.clamp(0.0, 1.0)
    gt = gt.clamp(0.0, 1.0)
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_type", type=str, required=True,
                        choices=["nomask", "predmask", "gtmask"],
                        help="nomask: 3ch / predmask: 4ch(pred_mask) / gtmask: 4ch(gt_mask)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="実験名 (未指定時は mask_type から自動生成)")
    parser.add_argument("--total_iters", type=int, default=200000)
    parser.add_argument("--val_freq", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--resume_weight", type=str, default="",
                        help="重みのみ復元 (optimizer/schedulerはリセット)")
    parser.add_argument("--resume_state", type=str, default="",
                        help="全状態を復元 (optimizer/schedulerも含む)")
    args = parser.parse_args()

    # 実験名
    if args.exp_name is None:
        args.exp_name = f"MambaIR_5stain_{args.mask_type.capitalize()}"

    device = torch.device(f"cuda:{args.device_id}")
    print(f"Training on {device}  exp={args.exp_name}  mask={args.mask_type}")

    wandb.init(project="Kuzushiji_Restoration", name=args.exp_name, config=vars(args))

    # データパス
    base = Path("../../data/hiragana_fulldataset_5stain")

    mask_type = args.mask_type
    if mask_type == "nomask":
        train_mask_dir = None
        val_mask_dir   = None
        in_chans = 3
    elif mask_type == "predmask":
        train_mask_dir = base / "pred_mask" / "train"
        val_mask_dir   = base / "pred_mask" / "val"
        in_chans = 4
    else:  # gtmask
        train_mask_dir = base / "gt_mask" / "train"
        val_mask_dir   = base / "gt_mask" / "val"
        in_chans = 4

    train_ds = MambaDataset(
        base / "lq" / "train", base / "gt" / "train",
        mask_dir=train_mask_dir, mode="train"
    )
    val_ds = MambaDataset(
        base / "lq" / "val", base / "gt" / "val",
        mask_dir=val_mask_dir, mode="val"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    # モデル (NAFNetと揃えた軽量設定)
    model = MambaIR(
        upscale=1, in_chans=in_chans, out_chans=3,
        img_size=128, embed_dim=64, depths=(4, 4, 4, 4), d_state=16
    )

    current_iter = 0
    best_psnr = 0.0
    checkpoint = None

    if args.resume_state:
        print(f"Loading full state from {args.resume_state}")
        checkpoint = torch.load(args.resume_state, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    elif args.resume_weight:
        print(f"Loading weights from {args.resume_weight}")
        checkpoint = torch.load(args.resume_weight, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=2e-4, betas=(0.9, 0.9), weight_decay=0.0
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iters, eta_min=1e-7
    )

    if args.resume_state and checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        current_iter = checkpoint.get("iter", 0)
        best_psnr    = checkpoint.get("best_psnr", 0.0)
        print(f"Resumed from iter {current_iter}, best_psnr={best_psnr:.4f}")

    criterion_charb  = CharbonnierLoss(eps=1e-3).to(device)
    criterion_percep = lpips.LPIPS(net="vgg").to(device)
    for p in criterion_percep.parameters():
        p.requires_grad = False

    save_dir = Path("experiments") / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    train_iter  = iter(train_loader)
    train_loss  = 0.0
    pbar = tqdm(total=args.total_iters, initial=current_iter, desc=args.exp_name)

    while current_iter < args.total_iters:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        current_iter += 1
        inputs = batch["input"].to(device)
        gts    = batch["gt"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds      = model(inputs)
            loss_charb = criterion_charb(preds, gts)
            loss_percep = criterion_percep(
                preds.float() * 2 - 1, gts.float() * 2 - 1
            ).mean()
            loss = loss_charb + 0.1 * loss_percep

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        if current_iter % 100 == 0:
            wandb.log({
                "train/loss": train_loss / 100,
                "train/lr":   scheduler.get_last_lr()[0],
                "iter":       current_iter,
            })
            train_loss = 0.0

        # Validation
        if current_iter % args.val_freq == 0:
            model.eval()
            val_psnr = 0.0
            with torch.no_grad():
                for vb in tqdm(val_loader, desc=f"Val@{current_iter}", leave=False):
                    v_in  = vb["input"].to(device)
                    v_gt  = vb["gt"].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v_pred = model(v_in)
                    for p_out, g_out in zip(v_pred, v_gt):
                        val_psnr += calculate_psnr(p_out, g_out)

            val_psnr /= len(val_ds)
            print(f"\nIter {current_iter}: Val PSNR {val_psnr:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            wandb.log({"val/psnr": val_psnr, "iter": current_iter})

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "iter":       current_iter,
                    "best_psnr":  best_psnr,
                    "mask_type":  mask_type,
                    "in_chans":   in_chans,
                }, save_dir / "best_model.pth")
                print(f"Saved best model PSNR={best_psnr:.4f}")

    # 最終保存
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "iter":      current_iter,
        "best_psnr": best_psnr,
        "mask_type": mask_type,
        "in_chans":  in_chans,
    }, save_dir / "latest_model.pth")

    pbar.close()
    wandb.finish()
    print(f"Training finished! best_psnr={best_psnr:.4f}")


if __name__ == "__main__":
    main()
