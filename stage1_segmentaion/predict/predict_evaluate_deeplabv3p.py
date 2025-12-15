#!/usr/bin/env python3
import os
import logging
import argparse
import re
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from tqdm import tqdm
import ssl

# SSL証明書エラー回避
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # ファイル名マッチング用の正規表現
        prefix_pattern = re.compile(r"^type\d+_sev[\d\.]+_")
        suffix_pattern = re.compile(r"_(Ghosting|Missing|Stain|Scratch|Transparent_Stain)$")

        def get_clean_stem(path_obj):
            stem = path_obj.stem
            stem = prefix_pattern.sub("", stem)
            stem = suffix_pattern.sub("", stem)
            return stem

        imgs = sorted([p for p in self.image_dir.iterdir() if p.is_file()])
        masks = {p.stem: p for p in self.mask_dir.iterdir() if p.is_file()}
        
        self.pairs = []
        for img in imgs:
            # 1. 完全一致またはクリーニング後のステムでマッチング
            key = get_clean_stem(img)
            if key in masks:
                self.pairs.append((img, masks[key]))
                continue
            
            # 2. サフィックス除去 (_Missing, _lq など) してマッチング (学習スクリプトのロジックに準拠)
            stem_clean = re.sub(r'(_Missing|_lq|_input)$', '', img.stem, flags=re.IGNORECASE)
            if stem_clean in masks:
                self.pairs.append((img, masks[stem_clean]))

        if not self.pairs:
            raise FileNotFoundError(f"No image/mask pairs found in {image_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img_path.name, img, mask


def build_transforms(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def format_name(label: str) -> str:
    return ''.join(part.capitalize() for part in re.split("[_\\-]", label) if part)


def format_encoder_name(enc: str) -> str:
    parts = re.split("[_\\-]", enc)
    formatted = []
    for part in parts:
        lower = part.lower()
        if lower.startswith("resnet"):
            formatted.append("ResNet" + part[len("resnet"):])
        else:
            formatted.append(part.capitalize())
    return "".join(formatted)


def soft_iou(pred, target, eps=1e-6):
    inter = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def hard_iou(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum((1, 2, 3))
    union = pred_bin.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def f_measure(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum((1, 2, 3))
    precision = inter / (pred_bin.sum((1, 2, 3)) + eps)
    recall = inter / (target.sum((1, 2, 3)) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.mean().item()


def mae_metric(pred, target):
    return torch.abs(pred - target).mean().item()


def build_model(encoder_name, encoder_weights, encoder_depth=5):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights or None,
        encoder_depth=encoder_depth,
        in_channels=3,
        classes=1
    )
    return model.to(DEVICE)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict and evaluate DeepLabV3+ soft masks")
    parser.add_argument("--data-dir", type=str, default="Kuzushiji_Restoration/datasets/dataset_final_hiragana")
    
    # 重みファイルのパス
    parser.add_argument("--weights", type=str, default="Kuzushiji_Restoration/experiments/segmentation/baseline/deeplabv3p_baseline.pth")
    
    # エンコーダー設定 (学習時と同じにする)
    parser.add_argument("--encoder-name", type=str, default="se_resnext50_32x4d")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--encoder-depth", type=int, default=5)
    
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    
    # 出力先ディレクトリ
    parser.add_argument("--output-dir", type=str, default="Kuzushiji_Restoration/experiments/segmentation/baseline/deeplabv3p")
    
    parser.add_argument("--wandb-project", type=str, default="Kuzushiji_Restoration")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    use_wandb = not args.no_wandb
    run_name = f"EvalDeepLabV3p_{format_encoder_name(args.encoder_name)}"
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args))
        columns = ["id", "image", "ground_truth", "prediction"]
        test_table = wandb.Table(columns=columns)

    transform = build_transforms(args.image_size)
    dataset = EvalDataset(
        os.path.join(args.data_dir, "lq5/test"),
        os.path.join(args.data_dir, "mask_gt/test"),
        transform
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.encoder_name, args.encoder_weights, args.encoder_depth)
    state = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    mae_list, soft_iou_list, hard_iou_list, f1_list = [], [], [], []

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for names, imgs, masks in tqdm(loader, desc="Eval"):
            imgs_gpu = imgs.to(DEVICE, dtype=torch.float)
            masks_gpu = masks.to(DEVICE, dtype=torch.float)
            outputs = model(imgs_gpu)
            probs = torch.sigmoid(outputs)

            mae_list.append(mae_metric(probs, masks_gpu))
            soft_iou_list.append(soft_iou(probs, masks_gpu))
            hard_iou_list.append(hard_iou(probs, masks_gpu))
            f1_list.append(f_measure(probs, masks_gpu))

            probs_np = probs.cpu().numpy()
            imgs_np = imgs.numpy()
            masks_np = masks.numpy()

            for i, name in enumerate(names):
                prob = probs_np[i][0]
                mask_img = (prob * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(mask_img).save(Path(args.output_dir) / Path(name).with_suffix(".jpg"))

                if use_wandb and len(test_table.data) < 32:
                    img_vis = imgs_np[i].transpose(1, 2, 0)
                    img_vis = (img_vis * std + mean).clip(0, 1)
                    img_vis = (img_vis * 255).astype(np.uint8)
                    gt_vis = (masks_np[i][0] * 255).clip(0, 255).astype(np.uint8)
                    pred_vis = mask_img
                    test_table.add_data(
                        name,
                        wandb.Image(img_vis),
                        wandb.Image(gt_vis),
                        wandb.Image(pred_vis)
                    )

    metrics = {
        "MAE": float(np.mean(mae_list)),
        "SoftIoU": float(np.mean(soft_iou_list)),
        "HardIoU": float(np.mean(hard_iou_list)),
        "F1": float(np.mean(f1_list))
    }
    logging.info("Evaluation results: %s", metrics)
    if use_wandb:
        wandb.log(metrics)
        wandb.log({"evaluation_samples": test_table})
        wandb.finish()

if __name__ == "__main__":
    main()