#!/usr/bin/env python3
"""
簡易 Restormer 学習スクリプト（マスクを使わない LQ -> GT の復元学習）

使い方の例:
  python train_restormer_nomask.py \
    --data-dir dataset_final_hiragana \
    --train-lq lq/train --train-gt gt/train \
    --val-lq lq/val --val-gt gt/val \
    --epochs 50 --batch-size 8 --image-size 256 \
    --save-path restormer_nomask_best.pth

このスクリプトはリポジトリ内の Restormer 実装
`basicsr.models.archs.restormer_arch.Restormer` を利用します。
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
import re
import wandb

# NOTE: basicsr/Restormer と psnr/ssim はインポートが重たいので遅延インポートします
Restormer = None
calculate_psnr = None
calculate_ssim = None


def list_image_files(d, allowed_exts={'.jpg', '.jpeg', '.png'}):
    """ディレクトリ内の画像ファイルリスト（拡張子フィルタ）を返す"""
    if not os.path.isdir(d):
        return []
    return sorted([f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in allowed_exts])


def make_stem_map(files):
    """stem -> [filenames] のマップを作る（拡張子を除く stem をキーに）"""
    m = {}
    for f in files:
        stem = os.path.splitext(f)[0]
        m.setdefault(stem, []).append(f)
    return m


def longest_common_prefix_len(a: str, b: str) -> int:
    # os.path.commonprefix を使って共通接頭辞の長さを返す
    return len(os.path.commonprefix([a, b]))


class PairedImageNoMaskDataset(Dataset):
    """lq と gt をベース名でペアにする簡易データセット（マスクなし）"""

    def __init__(self, lq_dir, gt_dir, transform=None, crop_size=None):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.transform = transform
        # crop_size を保持（None ならリサイズのみ）
        self.crop_size = crop_size

        if not os.path.isdir(lq_dir) or not os.path.isdir(gt_dir):
            logging.error(f"ディレクトリが見つかりません。 LQ: {lq_dir}, GT: {gt_dir}")
            self.pairs = []
            return

        # robust & efficient pairing:
        # 1) normalize stems by removing common suffixes (e.g. _alpha30, _prediction)
        def normalize(stem):
            s = re.sub(r'(_alpha\d+|_prediction|_pred|_restored|_mask)$', '', stem)
            return s

        lq_files = list_image_files(lq_dir)
        gt_files = list_image_files(gt_dir)
        # maps: original stem -> filename list
        lq_map = make_stem_map(lq_files)
        gt_map = make_stem_map(gt_files)

        # normalized maps: norm -> list of filenames
        lq_norm = {}
        for stem, fnames in lq_map.items():
            n = normalize(stem)
            lq_norm.setdefault(n, []).extend(fnames)
        gt_norm = {}
        for stem, fnames in gt_map.items():
            n = normalize(stem)
            gt_norm.setdefault(n, []).extend(fnames)

        pairs = []
        used_lq = set()

        # Step A: exact original stem matches
        for stem in sorted(set(lq_map.keys()) & set(gt_map.keys())):
            pairs.append((lq_map[stem][0], gt_map[stem][0]))
            used_lq.add(lq_map[stem][0])

        # Step B: exact normalized stem matches (handles suffix differences)
        for n in sorted(set(lq_norm.keys()) & set(gt_norm.keys())):
            # for each normalized key, pair in order
            lq_list = [f for f in lq_norm[n] if f not in used_lq]
            gt_list = gt_norm[n]
            for i, gt_fname in enumerate(gt_list):
                if i < len(lq_list):
                    pairs.append((lq_list[i], gt_fname))
                    used_lq.add(lq_list[i])

        # Step C: group-by-first-token and greedy longest-prefix within group (fast)
        # build groups by first token (e.g. 'U+3042')
        def first_token(stem):
            return stem.split('_', 1)[0]

        lq_groups = {}
        for stem, fnames in lq_map.items():
            tok = first_token(stem)
            lq_groups.setdefault(tok, []).append((stem, fnames[0]))

        # remaining GT stems to match
        paired_gt_set = set(p[1] for p in pairs)
        remaining_gt = [(stem, gt_map[stem][0]) for stem in gt_map.keys() if gt_map[stem][0] not in paired_gt_set]
        for gt_stem, gt_fname in remaining_gt:
            tok = first_token(gt_stem)
            candidates = lq_groups.get(tok, [])
            best = None; best_len = 0
            for lq_stem, lq_fname in candidates:
                if lq_fname in used_lq:
                    continue
                lcp = longest_common_prefix_len(gt_stem, lq_stem)
                if lcp > best_len and (lcp >= max(8, len(gt_stem)//2)):
                    best_len = lcp
                    best = lq_fname
            if best is not None:
                pairs.append((best, gt_fname))
                used_lq.add(best)

        # Final fallback: substring match on normalized stems (cheap, rare)
        paired_gt_set = set(p[1] for p in pairs)
        for gt_stem in gt_map.keys():
            gt_fname = gt_map[gt_stem][0]
            if gt_fname in paired_gt_set:
                continue
            gn = normalize(gt_stem)
            # try to find any lq_norm key that contains gn as substring
            found = False
            for ln, lfnames in lq_norm.items():
                if gn in ln:
                    for candidate in lfnames:
                        if candidate not in used_lq:
                            pairs.append((candidate, gt_fname))
                            used_lq.add(candidate)
                            found = True
                            break
                if found:
                    break

        self.pairs = pairs
        if len(self.pairs) == 0:
            logging.error(f"共通するファイルが見つかりません: {lq_dir}, {gt_dir}")
        else:
            logging.info(f'Paired {len(self.pairs)} files between {lq_dir} and {gt_dir}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lq_fname, gt_fname = self.pairs[idx]
        lq_path = os.path.join(self.lq_dir, lq_fname)
        gt_path = os.path.join(self.gt_dir, gt_fname)

        # PIL で読み込み -> 先にクロップ/リサイズしてからテンソル化
        lq_img = Image.open(lq_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        if self.crop_size is not None:
            cw = self.crop_size
            lw, lh = lq_img.size
            gw, gh = gt_img.size
            # 両方同サイズであることを期待するが、安全側で最小を使う
            w = min(lw, gw)
            h = min(lh, gh)
            if w >= cw and h >= cw:
                # ランダムクロップ
                left = np.random.randint(0, w - cw + 1)
                top = np.random.randint(0, h - cw + 1)
                lq_img = lq_img.crop((left, top, left + cw, top + cw))
                gt_img = gt_img.crop((left, top, left + cw, top + cw))
            else:
                # 小さければリサイズして返す
                lq_img = lq_img.resize((cw, cw), Image.BICUBIC)
                gt_img = gt_img.resize((cw, cw), Image.BICUBIC)

        lq = np.array(lq_img, dtype=np.float32) / 255.0
        gt = np.array(gt_img, dtype=np.float32) / 255.0

        # HWC -> CHW
        lq = torch.from_numpy(lq.transpose(2, 0, 1)).float()
        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float()

        return lq, gt


def collate_fn(batch):
    lqs = [b[0] for b in batch]
    gts = [b[1] for b in batch]
    lqs = torch.stack(lqs, dim=0)
    gts = torch.stack(gts, dim=0)
    return lqs, gts


def validate(model, val_loader, device, use_amp=False):
    # 遅延インポート: validate 内で psnr/ssim を読み込む（import が重い環境を回避）
    global calculate_psnr, calculate_ssim
    if calculate_psnr is None or calculate_ssim is None:
        from basicsr.metrics.psnr_ssim import calculate_psnr as _psnr, calculate_ssim as _ssim
        calculate_psnr, calculate_ssim = _psnr, _ssim

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    with torch.no_grad():
        # support limiting validation batches for quick debug
        max_val_batches = getattr(model, '_max_val_batches', None)
        if max_val_batches is not None and max_val_batches > 0:
            total = min(len(val_loader), max_val_batches)
        else:
            total = len(val_loader)
        for i, (lq, gt) in enumerate(tqdm(val_loader, desc='Val', leave=False, total=total)):
            if max_val_batches is not None and max_val_batches > 0 and i >= max_val_batches:
                break
            lq = lq.to(device)
            gt = gt.to(device)
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    out = model(lq)[0]
            else:
                out = model(lq)[0]
            # clamp
            out = torch.clamp(out, 0.0, 1.0)
            for i in range(out.size(0)):
                pred = out[i].unsqueeze(0)
                target = gt[i].unsqueeze(0)
                psnr = calculate_psnr(pred, target, crop_border=0, input_order='CHW', test_y_channel=False)
                ssim = calculate_ssim((pred.squeeze(0).permute(1,2,0).cpu().numpy()), (target.squeeze(0).permute(1,2,0).cpu().numpy()), crop_border=0, input_order='HWC', test_y_channel=False)
                total_psnr += psnr
                total_ssim += ssim
                count += 1
    if count == 0:
        return None, None
    return total_psnr / count, total_ssim / count


def train(args):
    # prepare logging: ensure directory exists and append to log file
    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(args.log_path, mode='a'), logging.StreamHandler()])
    logging.info('Start Restormer training (no mask)')

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=vars(args),
            name=f"Restormer_NoMask_{time.strftime('%Y%m%d_%H%M%S')}"
        )

    # --- debug probes: timestamped progress logs to find hang point ---
    logging.info('DEBUG: about to build datasets')
    t0 = time.time()
    try:
        train_dataset = PairedImageNoMaskDataset(os.path.join(args.data_dir, args.train_lq), os.path.join(args.data_dir, args.train_gt), crop_size=args.image_size)
        logging.info(f'DEBUG: train dataset built in {time.time()-t0:.3f}s, pairs={len(train_dataset)}')
    except Exception as e:
        logging.exception('DEBUG: failed building train_dataset')
        raise

    t0 = time.time()
    try:
        val_dataset = PairedImageNoMaskDataset(os.path.join(args.data_dir, args.val_lq), os.path.join(args.data_dir, args.val_gt), crop_size=args.image_size)
        logging.info(f'DEBUG: val dataset built in {time.time()-t0:.3f}s, pairs={len(val_dataset)}')
    except Exception as e:
        logging.exception('DEBUG: failed building val_dataset')
        raise
    # --- end debug probes ---

    if len(train_dataset) == 0:
        logging.error('Train dataset is empty. Check paths.')
        return
    if len(val_dataset) == 0:
        logging.warning('Val dataset is empty. Validation will be skipped.')

    # Log dataset sizes (pairs)
    logging.info(f'Train pairs: {len(train_dataset)}')
    logging.info(f'Val pairs: {len(val_dataset)}')

    logging.info('DEBUG: about to create DataLoaders')
    t0 = time.time()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    logging.info(f'DEBUG: train DataLoader created in {time.time()-t0:.3f}s')
    if len(val_dataset) > 0:
        t0 = time.time()
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
        logging.info(f'DEBUG: val DataLoader created in {time.time()-t0:.3f}s')
    else:
        val_loader = None

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logging.info(f'Using device: {device}')

    # 遅延インポート: Restormer をここで読み込む（ファイル import 時の遅延を回避）
    logging.info('DEBUG: about to import Restormer and instantiate model')
    global Restormer
    if Restormer is None:
        from basicsr.models.archs.restormer_arch import Restormer as _Restormer
        Restormer = _Restormer
    logging.info('DEBUG: Restormer imported')
    model = Restormer(inp_channels=3, out_channels=3)
    logging.info('DEBUG: model instantiated')
    model = model.to(device)
    logging.info('DEBUG: model moved to device')

    # Save initial weights (helps debugging when no later checkpoints are produced)
    try:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        init_path = os.path.splitext(args.save_path)[0] + '.init.pth'
        torch.save(model.state_dict(), init_path)
        logging.info(f'Initial model state saved to {init_path}')
    except Exception:
        logging.exception('Failed to save initial model state')

    # optionally load pretrained weights for fine-tuning
    if getattr(args, 'pretrained', None):
        try:
            state = torch.load(args.pretrained, map_location='cpu')
            # allow checkpoints that wrap state_dict under a key
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            # remove possible "module." prefixes
            if isinstance(state, dict):
                try:
                    model.load_state_dict(state, strict=bool(args.strict_load))
                    logging.info(f'Loaded pretrained state_dict from {args.pretrained} (strict={args.strict_load})')
                except Exception:
                    # try stripping module. prefixes and load with strict=False
                    new_state = {k.replace('module.', ''): v for k, v in state.items()}
                    model.load_state_dict(new_state, strict=False)
                    logging.info(f'Loaded pretrained state_dict from {args.pretrained} with relaxed strict=False')
            else:
                logging.warning('Pretrained file does not contain a state_dict-like dict')
        except Exception:
            logging.exception(f'Failed to load pretrained weights from {args.pretrained}')

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Prepare AMP scaler if requested
    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    best_psnr = -1.0

    # prepare metrics CSV
    metrics_csv = args.metrics_csv or (os.path.splitext(args.log_path)[0] + '_metrics.csv')
    metrics_dir = os.path.dirname(metrics_csv)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_csv, 'w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['timestamp', 'epoch', 'train_loss', 'val_psnr', 'val_ssim'])

    try:
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
            # optional limit for fast debug runs
            max_train_batches = int(getattr(args, 'max_train_batches', 0)) or None
            if max_train_batches is not None and max_train_batches > 0:
                train_total = min(len(train_loader), max_train_batches)
            else:
                train_total = len(train_loader)
            for batch_idx, (lq, gt) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False, total=train_total)):
                if max_train_batches is not None and max_train_batches > 0 and batch_idx >= max_train_batches:
                    break
                lq = lq.to(device)
                gt = gt.to(device)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = model(lq)[0]
                        out = torch.clamp(out, 0.0, 1.0)
                        loss = criterion(out, gt)
                        loss = loss / accum_steps
                    scaler.scale(loss).backward()
                else:
                    out = model(lq)[0]
                    out = torch.clamp(out, 0.0, 1.0)
                    loss = criterion(out, gt)
                    loss = loss / accum_steps
                    loss.backward()

                # step optimizer every accum_steps
                if (batch_idx + 1) % accum_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                # accumulate unscaled loss for reporting
                try:
                    running_loss += (loss.item() * accum_steps) * lq.size(0)
                except Exception:
                    running_loss += 0.0

            # handle leftover gradients if dataset size not divisible by accum_steps
            if (batch_idx + 1) % accum_steps != 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            avg_loss = running_loss / len(train_loader.dataset)
            logging.info(f'Epoch {epoch+1}: Train Loss: {avg_loss:.6f}')

            # Validation + metrics
            val_psnr, val_ssim = None, None
            if val_loader is not None:
                # pass max_val_batches via a temporary attribute on model
                max_val_batches = int(getattr(args, 'max_val_batches', 0)) or None
                setattr(model, '_max_val_batches', max_val_batches)
                val_psnr, val_ssim = validate(model, val_loader, device, use_amp=(scaler is not None))
                # cleanup
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
                # save best
                if val_psnr is not None and val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(model.state_dict(), args.save_path)
                    logging.info(f'Best model saved to {args.save_path} (PSNR: {best_psnr:.4f})')

            # append metrics to CSV
            try:
                with open(metrics_csv, 'a', newline='') as mf:
                    writer = csv.writer(mf)
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), epoch+1, f'{avg_loss:.6f}', f'{val_psnr if val_psnr is not None else "":}', f'{val_ssim if val_ssim is not None else "":}'])
            except Exception:
                logging.exception('Failed to write metrics CSV')

            # periodic save every N epochs
            if (epoch + 1) % args.save_every == 0:
                periodic_dir = os.path.dirname(args.save_path)
                if periodic_dir:
                    os.makedirs(periodic_dir, exist_ok=True)
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
        # Ensure log handlers flush and close
        logging.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Restormer (no mask)')
    # Defaults updated to match user's existing workspace structure
    parser.add_argument('--data-dir', type=str, default='dataset_final_hiragana', help='dataset root')
    parser.add_argument('--train-lq', type=str, default='lq_3stage/train', help='train lq relative to data-dir')
    parser.add_argument('--train-gt', type=str, default='gt/train', help='train gt relative to data-dir')
    parser.add_argument('--val-lq', type=str, default='lq_3stage/val', help='val lq relative to data-dir')
    parser.add_argument('--val-gt', type=str, default='gt/val', help='val gt relative to data-dir')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--save-path', type=str, default='restormer_nomask_best.pth')
    parser.add_argument('--log-path', type=str, default='restormer_nomask.log')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true', help='force cpu')
    parser.add_argument('--save-every', type=int, default=5, help='save checkpoint every N epochs')
    parser.add_argument('--metrics-csv', type=str, default='', help='path to CSV file to append metrics')
    parser.add_argument('--use-amp', action='store_true', help='enable AMP (mixed precision)')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained state_dict to load for fine-tuning')
    parser.add_argument('--strict-load', action='store_true', help='use strict loading when loading pretrained state_dict')
    parser.add_argument('--accum-steps', type=int, default=1, help='gradient accumulation steps to simulate larger batch')
    parser.add_argument('--max-train-batches', type=int, default=0, help='limit number of training batches per epoch (0 = all)')
    parser.add_argument('--max-val-batches', type=int, default=0, help='limit number of validation batches per epoch (0 = all)')
    
    # WandB arguments
    parser.add_argument('--wandb-project', type=str, default='Kuzushiji_Restoration', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb-group', type=str, default='stage2_restoration', help='wandb group name')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
