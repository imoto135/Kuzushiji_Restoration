"""
train.py — Adaptive Joint End-to-End Training (UNet++ + NAFNet)

Improvements over v1:
  - Learnable temperature τ in JointRestorationNet (mask sharpness)
  - Progressive unfreezing: Phase 1 (NAFNet warmup) → Phase 2 (soft unfreeze)
    → Phase 3 (full end-to-end)
  - λ_seg dynamic scheduling: linearly decays from lambda_seg_start to
    lambda_seg_end over training epochs
  - Temperature τ logged to WandB

Usage:
  cd /home/imoto/Kuzushiji_Restoration/models/joint
  python train.py --config options/Kuzushiji/joint_adaptive_mask.yml
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp
import yaml

# ── local imports ──────────────────────────────────────────────────────────
_THIS   = Path(__file__).resolve().parent
_MODELS = _THIS.parent
_NAFNET = _MODELS / "nafnet"
for _p in [str(_NAFNET)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint_model   import JointRestorationNet
from joint_dataset import KuzushijiJointDataset, get_transforms

# NAFNet perceptual loss (reuse BasicSR's implementation)
from basicsr.models.losses.losses import PerceptualLoss, CharbonnierLoss


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# PSNR helper
# ─────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Computes mean PSNR over a batch (tensors in [0,1])."""
    with torch.no_grad():
        mse = torch.mean((pred.clamp(0, 1) - target) ** 2, dim=[1, 2, 3])
        psnr_batch = -10.0 * torch.log10(mse + 1e-8)
    return psnr_batch.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper (for mask quality tracking)
# ─────────────────────────────────────────────────────────────────────────────

def iou_score(pred_mask: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """pred_mask is already sigmoid output (not logits)."""
    pred = (pred_mask > 0.5).float()
    inter = (pred * target).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) - inter
    return ((inter + eps) / (union + eps)).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# λ_seg scheduler
# ─────────────────────────────────────────────────────────────────────────────

def get_lambda_seg(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """Linearly decay λ_seg from `start` to `end` over all training epochs."""
    if total_epochs <= 1:
        return start
    t = (epoch - 1) / (total_epochs - 1)          # 0.0 → 1.0
    return start + t * (end - start)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer factory — separate param group for UNet++ (for phase lr control)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: JointRestorationNet, lr_nafnet: float, lr_unetpp: float,
                    weight_decay: float) -> torch.optim.Optimizer:
    """
    Build optimizer with two param groups:
        group 0: NAFNet + temperature τ
        group 1: UNet++

    Separate groups allow setting different lr for each network
    during progressive unfreezing (Phase 2).
    """
    nafnet_params = list(model.nafnet.parameters())
    temp_params   = [model.log_temp] if isinstance(model.log_temp, nn.Parameter) else []
    unetpp_params = list(model.unetpp.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": nafnet_params + temp_params, "lr": lr_nafnet},
            {"params": unetpp_params,               "lr": lr_unetpp},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.9),
    )
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scaler,
    charb_loss, percep_loss, dice_loss, bce_loss,
    lambda_seg, lambda_percep, device
) -> dict:
    model.train()
    totals = dict(total=0., restore=0., charb=0., percep=0., seg=0., temp=0.)
    count  = 0

    for batch in tqdm(loader, leave=False, desc="Train"):
        lq      = batch["lq"].to(device)
        gt      = batch["gt"].to(device)
        gt_mask = batch["gt_mask"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            restored, mask, temp = model(lq)

            # ── Restoration losses ──────────────────────────────────────────
            l_charb = charb_loss(restored, gt)

            if percep_loss is not None:
                l_percep, _ = percep_loss(restored, gt)
                l_percep    = l_percep if l_percep is not None else torch.tensor(0.)
            else:
                l_percep = torch.tensor(0., device=device)

            l_restore = l_charb + lambda_percep * l_percep

            # ── Segmentation losses (UNet++ guidance) ───────────────────────
            # F.binary_cross_entropy is unsafe in autocast; compute in float32
            l_dice = dice_loss(mask, gt_mask)

        # BCE must be computed outside autocast (unsafe with float16)
        with torch.cuda.amp.autocast(enabled=False):
            l_bce  = F.binary_cross_entropy(mask.float().clamp(0.0, 1.0), gt_mask.float().clamp(0.0, 1.0))

        l_seg   = 0.5 * l_dice + 0.5 * l_bce
        l_total = l_restore + lambda_seg * l_seg

        scaler.scale(l_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        B = lq.size(0)
        totals["total"]   += l_total.item()   * B
        totals["restore"] += l_restore.item() * B
        totals["charb"]   += l_charb.item()   * B
        totals["percep"]  += l_percep.item()  * B
        totals["seg"]     += l_seg.item()      * B
        totals["temp"]    += temp              * B
        count             += B

    return {k: v / count for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, charb_loss, dice_loss, lambda_seg, lpips_fn, device) -> dict:
    model.eval()
    totals = dict(psnr=0., lpips=0., iou=0., charb=0., seg=0., temp=0.)
    count  = 0

    for batch in tqdm(loader, leave=False, desc="Val"):
        lq      = batch["lq"].to(device)
        gt      = batch["gt"].to(device)
        gt_mask = batch["gt_mask"].to(device)

        with torch.cuda.amp.autocast():
            restored, mask, temp = model(lq)

        totals["psnr"] += psnr(restored, gt) * lq.size(0)
        with torch.no_grad():
            l_lpips = lpips_fn(restored * 2.0 - 1.0, gt * 2.0 - 1.0)
            totals["lpips"] += l_lpips.sum().item()
        totals["iou"]  += iou_score(mask, gt_mask) * lq.size(0)
        totals["charb"]+= charb_loss(restored, gt).item() * lq.size(0)
        l_dice = dice_loss(mask, gt_mask)
        with torch.cuda.amp.autocast(enabled=False):
            l_bce  = F.binary_cross_entropy(mask.float().clamp(0.0, 1.0), gt_mask.float().clamp(0.0, 1.0))
        totals["seg"]  += (0.5*l_dice + 0.5*l_bce).item() * lq.size(0)
        totals["temp"] += temp * lq.size(0)
        count          += lq.size(0)

    return {k: v / count for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Progressive phase management
# ─────────────────────────────────────────────────────────────────────────────

def get_current_phase(epoch: int, phase1_end: int, phase2_end: int) -> int:
    """Return current training phase (1, 2, or 3)."""
    if epoch <= phase1_end:
        return 1
    elif epoch <= phase2_end:
        return 2
    else:
        return 3


def apply_phase_settings(
    phase: int, prev_phase: int,
    model: JointRestorationNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    lr_nafnet: float, lr_unetpp: float, lr_unetpp_phase2: float,
    logger: logging.Logger,
) -> None:
    """Apply freeze/unfreeze and lr changes when transitioning between phases.

    IMPORTANT: scheduler.base_lrs must be updated together with optimizer.param_groups["lr"].
    Otherwise, the next scheduler.step() call will overwrite the manually set lr
    using the stale base_lr (e.g. 0.0 from Phase 1), causing UNet++ lr to stay near 0.
    """
    if phase == prev_phase:
        return  # No change needed

    logger.info(f"Phase transition: {prev_phase} → {phase}")

    if phase == 1:
        # NAFNet warmup: freeze UNet++ via requires_grad=False
        # Set lr=0 AND update base_lrs so scheduler doesn't use 0 as base in later phases
        model.freeze_unetpp()
        new_lrs = [lr_nafnet, 0.0]
        for pg, lr in zip(optimizer.param_groups, new_lrs):
            pg["lr"] = lr
            pg["initial_lr"] = lr
        scheduler.base_lrs = new_lrs
        logger.info(f"[Phase 1] UNet++ frozen. NAFNet lr={lr_nafnet:.2e}")

    elif phase == 2:
        # Soft unfreezing: UNet++ learns with low lr
        model.unfreeze_unetpp()
        new_lrs = [lr_nafnet, lr_unetpp_phase2]
        for pg, lr in zip(optimizer.param_groups, new_lrs):
            pg["lr"] = lr
            pg["initial_lr"] = lr
        scheduler.base_lrs = new_lrs
        logger.info(f"[Phase 2] UNet++ unfrozen at lr={lr_unetpp_phase2:.2e}.")

    elif phase == 3:
        # Full end-to-end: both networks at their nominal lr
        model.unfreeze_unetpp()
        new_lrs = [lr_nafnet, lr_unetpp]
        for pg, lr in zip(optimizer.param_groups, new_lrs):
            pg["lr"] = lr
            pg["initial_lr"] = lr
        scheduler.base_lrs = new_lrs
        logger.info(f"[Phase 3] Full end-to-end. UNet++ lr={lr_unetpp:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Adaptive Joint UNet++/NAFNet training")
    parser.add_argument("--config",   type=str,  required=True)
    parser.add_argument("--epochs",   type=int,  default=None, help="Override epochs")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)

    # ── Paths ──────────────────────────────────────────────────────────────
    data_base   = Path(cfg["data"]["base"])
    out_dir     = Path(cfg["output"]["dir"])
    ckpt_dir    = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "train.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {args.config}")

    # ── WandB ─────────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.get("wandb_project", "Kuzushiji_Restoration"),
                name=cfg.get("name", "joint_adaptive_mask"),
                config=cfg,
            )
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    img_size = cfg["data"].get("img_size", 128)
    train_ds = KuzushijiJointDataset(
        lq_dir   = data_base / "lq"      / "train",
        gt_dir   = data_base / "gt"      / "train",
        mask_dir = data_base / "gt_mask" / "train",
        mode     = "train", img_size=img_size,
    )
    val_ds = KuzushijiJointDataset(
        lq_dir   = data_base / "lq"      / "val",
        gt_dir   = data_base / "gt"      / "val",
        mask_dir = data_base / "gt_mask" / "val",
        mode     = "val",   img_size=img_size,
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"].get("batch_size", 16),
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 8),
        pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"].get("batch_size", 16),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 8),
        pin_memory=True, persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg   = cfg.get("model", {})
    learnable_temp = model_cfg.get("learnable_temp", True)
    temp_init      = model_cfg.get("temp_init", 1.0)

    model = JointRestorationNet(
        unetpp_pretrain = cfg["pretrain"].get("unetpp"),
        nafnet_pretrain = cfg["pretrain"].get("nafnet"),
        learnable_temp  = learnable_temp,
        temp_init       = temp_init,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Total parameters: {total_params:.2f} M")
    logger.info(f"Learnable temperature: {learnable_temp}  (τ_init={temp_init})")

    # ── Losses ────────────────────────────────────────────────────────────
    charb_loss   = CharbonnierLoss(loss_weight=1.0, reduction="mean").to(device)
    dice_loss_fn = smp.losses.DiceLoss(mode="binary")
    bce_loss_fn  = nn.BCELoss()   # kept for clarity (unused directly)

    lambda_percep = cfg["train"].get("lambda_percep", 0.1)

    # λ_seg schedule params
    lambda_seg_start = cfg["train"].get("lambda_seg_start", cfg["train"].get("lambda_seg", 0.1))
    lambda_seg_end   = cfg["train"].get("lambda_seg_end",   lambda_seg_start)

    percep_loss = None
    if cfg["train"].get("use_perceptual", True):
        percep_loss = PerceptualLoss(
            layer_weights={"conv5_4": 1.0},
            vgg_type="vgg19",
            use_input_norm=True,
            range_norm=False,
            perceptual_weight=1.0,
            style_weight=0,
        ).to(device)

    # ── Progressive unfreezing config ─────────────────────────────────────
    epochs     = args.epochs or cfg["train"].get("epochs", 100)
    phases_cfg = cfg.get("phases", {})

    phase1_end = phases_cfg.get("phase1_epochs", 0)           # 0 = skip phase 1
    phase2_end = phase1_end + phases_cfg.get("phase2_epochs", 0)

    lr_nafnet        = cfg["train"].get("lr", 1e-4)
    lr_unetpp        = cfg["train"].get("lr_unetpp", lr_nafnet)
    lr_unetpp_phase2 = cfg["train"].get("lr_unetpp_phase2", lr_unetpp * 0.1)

    logger.info(
        f"[Phase schedule] Phase1: ep 1-{phase1_end} (UNet++ frozen) | "
        f"Phase2: ep {phase1_end+1}-{phase2_end} (UNet++ lr={lr_unetpp_phase2:.2e}) | "
        f"Phase3: ep {phase2_end+1}-{epochs} (UNet++ lr={lr_unetpp:.2e})"
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        lr_nafnet=lr_nafnet,
        lr_unetpp=lr_unetpp,
        weight_decay=cfg["train"].get("weight_decay", 1e-5),
    )

    # ── Scheduler: CosineAnnealing over epochs (step once per epoch) ───────
    # T_max=epochs because scheduler.step() is called once per epoch, not per iter.
    # Using total_iters here would make lr almost frozen for the entire 100 epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )
    scaler = torch.cuda.amp.GradScaler()

    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # ── Apply Phase 1 settings at start ───────────────────────────────────
    current_phase  = 0  # sentinel — triggers apply_phase_settings on first epoch
    save_every     = cfg["train"].get("save_every", 10)
    es_patience    = cfg["train"].get("early_stop_patience", 5)
    es_counter     = 0
    best_lpips     = 100.0

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        # Phase management
        new_phase = get_current_phase(epoch, phase1_end, phase2_end)
        apply_phase_settings(
            phase=new_phase, prev_phase=current_phase,
            model=model, optimizer=optimizer,
            scheduler=scheduler,
            lr_nafnet=lr_nafnet, lr_unetpp=lr_unetpp,
            lr_unetpp_phase2=lr_unetpp_phase2,
            logger=logger,
        )
        current_phase = new_phase

        # λ_seg dynamic scheduling
        lambda_seg = get_lambda_seg(epoch, epochs, lambda_seg_start, lambda_seg_end)

        logger.info(
            f"═══ Epoch {epoch}/{epochs} | Phase={current_phase} | "
            f"λ_seg={lambda_seg:.4f} ═══"
        )

        train_logs = train_one_epoch(
            model, train_loader, optimizer, scaler,
            charb_loss, percep_loss, dice_loss_fn, bce_loss_fn,
            lambda_seg, lambda_percep, device,
        )
        scheduler.step()

        val_logs = validate(model, val_loader, charb_loss, dice_loss_fn, lambda_seg, lpips_fn, device)

        lr_nafnet_now = optimizer.param_groups[0]["lr"]
        lr_unetpp_now = optimizer.param_groups[1]["lr"]
        logger.info(
            f"Train total={train_logs['total']:.4f} "
            f"restore={train_logs['restore']:.4f} "
            f"seg={train_logs['seg']:.4f} "
            f"τ={train_logs['temp']:.4f} | "
            f"Val PSNR={val_logs['psnr']:.2f} dB LPIPS={val_logs['lpips']:.4f} IoU={val_logs['iou']:.4f} "
            f"τ={val_logs['temp']:.4f} | "
            f"lr_nafnet={lr_nafnet_now:.2e} lr_unetpp={lr_unetpp_now:.2e}"
        )

        if wandb_run:
            import wandb
            wandb.log({
                "phase":                 current_phase,
                "lambda_seg":            lambda_seg,
                "train/loss_total":      train_logs["total"],
                "train/loss_restore":    train_logs["restore"],
                "train/loss_charb":      train_logs["charb"],
                "train/loss_percep":     train_logs["percep"],
                "train/loss_seg":        train_logs["seg"],
                "train/temperature":     train_logs["temp"],
                "val/psnr":              val_logs["psnr"],
                "val/lpips":             val_logs["lpips"],
                "val/iou":               val_logs["iou"],
                "val/charb":             val_logs["charb"],
                "val/seg":               val_logs["seg"],
                "val/temperature":       val_logs["temp"],
                "lr/nafnet":             lr_nafnet_now,
                "lr/unetpp":             lr_unetpp_now,
            }, step=epoch)

        # ── Checkpointing ─────────────────────────────────────────────────
        if save_every > 0 and epoch % save_every == 0:
            ckpt = ckpt_dir / f"epoch_{epoch:04d}.pth"
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "best_lpips":  best_lpips,
                "phase":       current_phase,
            }, ckpt)
            logger.info(f"Checkpoint saved: {ckpt}")

        # ── Best model ─────────────────────────────────────────────────────
        if val_logs["lpips"] < best_lpips:
            best_lpips = val_logs["lpips"]
            es_counter = 0
            save_path  = out_dir / "best_model.pth"
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "best_lpips": best_lpips,
                "phase":      current_phase,
            }, save_path)
            logger.info(f"★ New best LPIPS={best_lpips:.4f} → saved to {save_path}")
            if wandb_run:
                import wandb
                wandb.run.summary["best_lpips"] = best_lpips
                wandb.run.summary["best_epoch"] = epoch
        else:
            es_counter += 1
            logger.info(f"No improvement. ES counter: {es_counter}/{es_patience}")
            if es_counter >= es_patience:
                if current_phase == 1:
                    logger.info("ES threshold reached in Phase 1. Skipping remaining epochs to start Phase 2.")
                    skipped_epochs = phase1_end - epoch
                    phase1_end = epoch
                    phase2_end -= skipped_epochs
                    es_counter = 0
                elif current_phase == 2:
                    logger.info("ES threshold reached in Phase 2. Skipping remaining epochs to start Phase 3.")
                    phase2_end = epoch
                    es_counter = 0
                else:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

    logger.info(f"Training complete. Best Val LPIPS: {best_lpips:.4f}")
    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
