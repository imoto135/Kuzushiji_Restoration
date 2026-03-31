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
class MambaStage2Dataset(Dataset):
    _PREFIX_RE = re.compile(r"^type\d+_sev[\d\.]+_")
    _SUFFIX_RE = re.compile(r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain|Abrasion|Transparent)$")

    def __init__(self, lq_dir, gt_dir, pred_mask_dir, img_size=128, mode="train"):
        self.lq_dir = Path(lq_dir)
        self.gt_dir = Path(gt_dir)
        self.mask_dir = Path(pred_mask_dir)
        self.mode = mode
        
        if mode == "train":
            self.transform = A.Compose(
                [A.Resize(img_size, img_size), A.Rotate(limit=10, p=0.3)],
                additional_targets={"gt": "image", "mask": "mask"}
            )
        else:
            self.transform = A.Compose(
                [A.Resize(img_size, img_size)],
                additional_targets={"gt": "image", "mask": "mask"}
            )
            
        self.triplets = self._build_triplets()

    @classmethod
    def _clean_stem(cls, path: Path) -> str:
        stem = path.stem
        stem = cls._PREFIX_RE.sub("", stem)
        stem = cls._SUFFIX_RE.sub("", stem)
        return stem

    def _index(self, directory: Path):
        return {
            self._clean_stem(f): f 
            for f in directory.iterdir() 
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

    def _build_triplets(self):
        lq_idx = self._index(self.lq_dir)
        gt_idx = self._index(self.gt_dir)
        mask_idx = self._index(self.mask_dir)
        common = set(lq_idx) & set(gt_idx) & set(mask_idx)
        return sorted([(lq_idx[k], gt_idx[k], mask_idx[k]) for k in common])

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        lq_p, gt_p, mask_p = self.triplets[idx]
        lq_img = np.array(Image.open(lq_p).convert("RGB"))
        gt_img = np.array(Image.open(gt_p).convert("RGB"))
        mask_m = np.array(Image.open(mask_p).convert("L"))

        transformed = self.transform(image=lq_img, gt=gt_img, mask=mask_m)
        lq_img = transformed["image"]
        gt_img = transformed["gt"]
        mask_m = transformed["mask"]

        # To [0, 1] tensors
        lq_t = torch.from_numpy(lq_img).permute(2, 0, 1).float() / 255.0
        gt_t = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_m).unsqueeze(0).float() / 255.0

        num_mask_channels = mask_t.shape[0]
        if num_mask_channels == 3:
             mask_t = mask_t[0:1] # ensure 1 channel
             
        # Concatenate for 4-channel input
        input_t = torch.cat([lq_t, mask_t], dim=0) # [4, 128, 128]
        
        return {"input": input_t, "gt": gt_t}

# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

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
# Training Script
# -----------------------------------------------------------------------------
def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_iters", type=int, default=200000)
    parser.add_argument("--val_freq", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device_id", type=int, default=1, help="GPU ID to use")
    parser.add_argument("--exp_name", type=str, default="MambaIR_Stage2_v2")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}")
    print(f"Training on {device}")

    # Set up wandb
    wandb.init(project="Kuzushiji_Restoration", name=args.exp_name, config=args)

    base = Path("../../data/full_padded")
    train_ds = MambaStage2Dataset(base/"lq/train", base/"gt/train", base/"pred_mask/train", mode="train")
    val_ds = MambaStage2Dataset(base/"lq/val", base/"gt/val", base/"pred_mask/val", mode="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Initialize MambaIR (Lighter configuration for fair comparison and speed)
    model = MambaIR(upscale=1, in_chans=4, out_chans=3, img_size=128, 
                    embed_dim=64, depths=(4, 4, 4, 4), d_state=16).to(device)

    # NAFNet setup: AdamW 2e-4 for stability
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.9), weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=1e-7)

    criterion_charb = CharbonnierLoss(eps=1e-6).to(device)
    criterion_percep = lpips.LPIPS(net="vgg").to(device)
    for p in criterion_percep.parameters():
        p.requires_grad = False

    scaler = torch.amp.GradScaler('cuda')
    best_psnr = 0.0
    save_dir = Path("experiments") / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    current_iter = 0
    train_iter = iter(train_loader)
    
    pbar = tqdm(total=args.total_iters, desc="Training")
    train_loss = 0.0

    while current_iter < args.total_iters:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        current_iter += 1
        
        inputs = batch["input"].to(device)
        gts = batch["gt"].to(device)

        optimizer.zero_grad()
        
        # Use bfloat16 strictly to trigger optimized Mamba hardware kernels
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            preds = model(inputs)
            
            loss_charb = criterion_charb(preds, gts)
            # LPIPS expects input in range [-1, 1], our data is [0, 1]
            loss_percep = criterion_percep(preds * 2 - 1, gts * 2 - 1).mean()
            
            # NAFNet setting: Charb: 1.0, Percep: 0.1
            loss = loss_charb + 0.1 * loss_percep

        scaler.scale(loss).backward()

        # Gradient clipping to prevent nan
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({"loss": loss.item()})
        
        # Log periodically
        if current_iter % 100 == 0:
            wandb.log({
                "train/loss": train_loss / 100,
                "train/lr": scheduler.get_last_lr()[0],
                "iter": current_iter
            })
            train_loss = 0.0

        # Validation
        if current_iter % args.val_freq == 0:
            model.eval()
            val_psnr = 0.0
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc=f"Iter {current_iter} [Val]", leave=False):
                    v_inputs = val_batch["input"].to(device)
                    v_gts = val_batch["gt"].to(device)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        v_preds = model(v_inputs)
                    for p_out, g_out in zip(v_preds, v_gts):
                        val_psnr += calculate_psnr(p_out, g_out)
                        
            val_psnr /= len(val_ds)
            
            print(f"\nIter {current_iter}: Val PSNR {val_psnr:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            wandb.log({
                "val/psnr": val_psnr,
                "iter": current_iter
            })

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                print(f"Saved best model with PSNR {best_psnr:.4f}")

    torch.save(model.state_dict(), save_dir / "latest_model.pth")
    wandb.finish()
    print("Training finished!")

if __name__ == "__main__":
    main()
