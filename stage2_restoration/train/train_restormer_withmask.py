#!/usr/bin/env python3
"""
Train Restormer using mask guidance: input is (RGB + mask) -> GT RGB.

Usage example:
  python train_restormer_withmask.py \
    --data-dir dataset_final_hiragana \
    --train-lq lq_random/train --train-gt gt/train --train-mask mask_gt/train \
    --val-lq lq_random/val --val-gt gt/val --val-mask mask_gt/val \
    --epochs 50 --batch-size 8 --image-size 128 --save-path restormer_withmask_best.pth

This is similar to train_restormer_nomask.py but concatenates the binary mask as an extra input channel.
"""
import os
import argparse
import logging
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import time
import wandb

from basicsr.models.archs.restormer_arch import Restormer
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def build_file_map(d, allowed_exts={'.jpg', '.jpeg', '.png'}, pref_order=['.jpg', '.jpeg', '.png']):
    m = {}
    if not os.path.isdir(d):
        return m
    for fname in os.listdir(d):
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in allowed_exts:
            continue
        if stem not in m:
            m[stem] = fname
        else:
            cur_ext = os.path.splitext(m[stem])[1].lower()
            if pref_order.index(ext) < pref_order.index(cur_ext):
                m[stem] = fname
    return m


class PairedImageMaskDataset(Dataset):
    """lq (RGB) + mask (L) paired with GT (RGB)."""
    def __init__(self, lq_dir, gt_dir, mask_dir, crop_size=None):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.mask_dir = mask_dir
        self.crop_size = crop_size

        if not os.path.isdir(lq_dir) or not os.path.isdir(gt_dir) or not os.path.isdir(mask_dir):
            logging.error(f"ディレクトリが見つかりません。 LQ: {lq_dir}, GT: {gt_dir}, MASK: {mask_dir}")
            self.pairs = []
            return

        lq_map = build_file_map(lq_dir)
        gt_map = build_file_map(gt_dir)
        mask_map = build_file_map(mask_dir)
        common = sorted(set(lq_map.keys()).intersection(set(gt_map.keys())).intersection(set(mask_map.keys())))
        self.pairs = [(lq_map[s], gt_map[s], mask_map[s]) for s in common]
        if len(self.pairs) == 0:
            logging.error(f"共通するファイルが見つかりません: {lq_dir}, {gt_dir}, {mask_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lq_fname, gt_fname, mask_fname = self.pairs[idx]
        lq_path = os.path.join(self.lq_dir, lq_fname)
        gt_path = os.path.join(self.gt_dir, gt_fname)
        mask_path = os.path.join(self.mask_dir, mask_fname)

        lq_img = Image.open(lq_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')

        if self.crop_size is not None:
            cw = self.crop_size
            lw, lh = lq_img.size
            gw, gh = gt_img.size
            mw, mh = mask_img.size
            w = min(lw, gw, mw)
            h = min(lh, gh, mh)
            if w >= cw and h >= cw:
                left = np.random.randint(0, w - cw + 1)
                top = np.random.randint(0, h - cw + 1)
                lq_img = lq_img.crop((left, top, left + cw, top + cw))
                gt_img = gt_img.crop((left, top, left + cw, top + cw))
                mask_img = mask_img.crop((left, top, left + cw, top + cw))
            else:
                lq_img = lq_img.resize((cw, cw), Image.BICUBIC)
                gt_img = gt_img.resize((cw, cw), Image.BICUBIC)
                mask_img = mask_img.resize((cw, cw), Image.NEAREST)

        lq = np.array(lq_img, dtype=np.float32) / 255.0
        gt = np.array(gt_img, dtype=np.float32) / 255.0
        mask = np.array(mask_img, dtype=np.uint8)
        # binarize mask: >127 -> 1
        mask = (mask > 127).astype(np.float32)

        # HWC -> CHW
        lq = torch.from_numpy(lq.transpose(2, 0, 1)).float()
        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask[None, ...]).float()

        # concatenate mask as extra channel on input
        inp = torch.cat([lq, mask], dim=0)
        return inp, gt


def collate_fn(batch):
    inps = [b[0] for b in batch]
    gts = [b[1] for b in batch]
    inps = torch.stack(inps, dim=0)
    gts = torch.stack(gts, dim=0)
    return inps, gts


def validate(model, val_loader, device, use_amp=False):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    with torch.no_grad():
        max_val_batches = getattr(model, '_max_val_batches', None)
        if max_val_batches is not None and max_val_batches > 0:
            total = min(len(val_loader), max_val_batches)
        else:
            total = len(val_loader)
        for i, (inp, gt) in enumerate(tqdm(val_loader, desc='Val', leave=False, total=total)):
            if max_val_batches is not None and max_val_batches > 0 and i >= max_val_batches:
                break
            inp = inp.to(device)
            gt = gt.to(device)
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    out = model(inp)[0]
            else:
                out = model(inp)[0]
            out = torch.clamp(out, 0.0, 1.0)
            for j in range(out.size(0)):
                pred = out[j].unsqueeze(0)
                target = gt[j].unsqueeze(0)
                psnr = calculate_psnr(pred, target, crop_border=0, input_order='CHW', test_y_channel=False)
                ssim = calculate_ssim((pred.squeeze(0).permute(1,2,0).cpu().numpy()), (target.squeeze(0).permute(1,2,0).cpu().numpy()), crop_border=0, input_order='HWC', test_y_channel=False)
                total_psnr += psnr
                total_ssim += ssim
                count += 1
    if count == 0:
        return None, None
    return total_psnr / count, total_ssim / count


def load_state_safe(model, path, strict=False):
    try:
        state = torch.load(path, map_location='cpu')
        sd = state.get('state_dict', state) if isinstance(state, dict) else state
        if isinstance(sd, dict):
            sd = { (k.replace('module.', '') if k.startswith('module.') else k): v for k, v in sd.items() }
            model.load_state_dict(sd, strict=strict)
            logging.info(f'Loaded pretrained weights from {path} (strict={strict})')
            return True
    except Exception as e:
        logging.warning(f'Failed to load pretrained weights: {e}')
    return False


def train(args):
    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(args.log_path, mode='a'), logging.StreamHandler()])
    logging.info('Start Restormer training (with mask guidance)')

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=vars(args),
            name=f"Restormer_WithMask_{time.strftime('%Y%m%d_%H%M%S')}"
        )

    # performance: enable cuDNN autotuner when using fixed-size inputs on GPU
    try:
        if torch.cuda.is_available() and not args.cpu:
            torch.backends.cudnn.benchmark = True
            logging.info('Enabled torch.backends.cudnn.benchmark for potential speedup')
    except Exception:
        pass

    train_dataset = PairedImageMaskDataset(os.path.join(args.data_dir, args.train_lq), os.path.join(args.data_dir, args.train_gt), os.path.join(args.data_dir, args.train_mask), crop_size=args.image_size)
    val_dataset = PairedImageMaskDataset(os.path.join(args.data_dir, args.val_lq), os.path.join(args.data_dir, args.val_gt), os.path.join(args.data_dir, args.val_mask), crop_size=args.image_size)

    if len(train_dataset) == 0:
        logging.error('Train dataset is empty. Check paths.')
        return
    if len(val_dataset) == 0:
        logging.warning('Val dataset is empty. Validation will be skipped.')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    ) if len(val_dataset) > 0 else None

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logging.info(f'Using device: {device}')

    # model expects 4-channel input (RGB + mask)
    model = Restormer(inp_channels=4, out_channels=3)
    model = model.to(device)

    # Save initial
    try:
        init_path = os.path.splitext(args.save_path)[0] + '.init.pth'
        torch.save(model.state_dict(), init_path)
        logging.info(f'Initial model state saved to {init_path}')
    except Exception:
        logging.exception('Failed to save initial model state')

    if args.pretrained:
        load_state_safe(model, args.pretrained, strict=False)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    best_psnr = -1.0

    metrics_csv = args.metrics_csv or (os.path.splitext(args.log_path)[0] + '_metrics.csv')
    with open(metrics_csv, 'w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['timestamp', 'epoch', 'train_loss', 'val_psnr', 'val_ssim'])

    try:
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
            max_train_batches = int(getattr(args, 'max_train_batches', 0)) or None
            if max_train_batches is not None and max_train_batches > 0:
                train_total = min(len(train_loader), max_train_batches)
            else:
                train_total = len(train_loader)
            for batch_idx, (inp, gt) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False, total=train_total)):
                if max_train_batches is not None and max_train_batches > 0 and batch_idx >= max_train_batches:
                    break
                inp = inp.to(device)
                gt = gt.to(device)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = model(inp)[0]
                        out = torch.clamp(out, 0.0, 1.0)
                        loss = criterion(out, gt) / accum_steps
                    scaler.scale(loss).backward()
                else:
                    out = model(inp)[0]
                    out = torch.clamp(out, 0.0, 1.0)
                    loss = criterion(out, gt) / accum_steps
                    loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                try:
                    running_loss += (loss.item() * accum_steps) * inp.size(0)
                except Exception:
                    running_loss += 0.0

            if (batch_idx + 1) % accum_steps != 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            avg_loss = running_loss / max(len(train_loader.dataset), 1)
            logging.info(f'Epoch {epoch+1}: Train Loss: {avg_loss:.6f}')

            val_psnr, val_ssim = None, None
            if val_loader is not None:
                setattr(model, '_max_val_batches', int(getattr(args, 'max_val_batches', 0)) or None)
                val_psnr, val_ssim = validate(model, val_loader, device, use_amp=(scaler is not None))
                if hasattr(model, '_max_val_batches'):
                    delattr(model, '_max_val_batches')
                logging.info(f'Epoch {epoch+1}: Val PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}')
                
                # WandB logging
                if not args.no_wandb:
                    wandb.log({
                        'train_loss': avg_loss,
                        'val_psnr': val_psnr,
                        'val_ssim': val_ssim,
                        'epoch': epoch + 1
                    })

                scheduler.step(val_psnr if val_psnr is not None else avg_loss)
                if val_psnr is not None and val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(model.state_dict(), args.save_path)
                    logging.info(f'Best model saved to {args.save_path} (PSNR: {best_psnr:.4f})')

            try:
                with open(metrics_csv, 'a', newline='') as mf:
                    writer = csv.writer(mf)
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), epoch+1, f'{avg_loss:.6f}', f'{val_psnr if val_psnr is not None else "":}', f'{val_ssim if val_ssim is not None else "":}'])
            except Exception:
                logging.exception('Failed to write metrics CSV')

            if (epoch + 1) % args.save_every == 0:
                periodic_path = f"{os.path.splitext(args.save_path)[0]}.epoch{epoch+1}.pth"
                torch.save(model.state_dict(), periodic_path)
                logging.info(f'Periodic checkpoint saved to {periodic_path} (epoch {epoch+1})')
    except KeyboardInterrupt:
        logging.info('Training interrupted by user')
        if not args.no_wandb:
            wandb.finish()
        raise
    except Exception:
        logging.exception('Unhandled exception during training')
        if not args.no_wandb:
            wandb.finish()
        raise
    finally:
        logging.info('Training finished (shutdown)')
        if not args.no_wandb:
            wandb.finish()
        logging.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Restormer with mask guidance')
    parser.add_argument('--data-dir', type=str, default='dataset_final_hiragana')
    parser.add_argument('--train-lq', type=str, default='lq_random/train')
    parser.add_argument('--train-gt', type=str, default='gt/train')
    parser.add_argument('--train-mask', type=str, default='mask_gt/train')
    parser.add_argument('--val-lq', type=str, default='lq_random/val')
    parser.add_argument('--val-gt', type=str, default='gt/val')
    parser.add_argument('--val-mask', type=str, default='mask_gt/val')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--save-path', type=str, default='restormer_withmask_best.pth')
    parser.add_argument('--log-path', type=str, default='restormer_withmask.log')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--metrics-csv', type=str, default='')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--persistent-workers', action='store_true', help='enable DataLoader.persistent_workers when num_workers>0')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch_factor (applies when num_workers>0)')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--strict-load', action='store_true')
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-val-batches', type=int, default=0)
    
    # WandB arguments
    parser.add_argument('--wandb-project', type=str, default='Kuzushiji_Restoration', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb-group', type=str, default='stage2_restoration', help='wandb group name')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')
    
    # num-workers defined earlier
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
