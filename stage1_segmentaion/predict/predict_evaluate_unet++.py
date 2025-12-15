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

# 【追加】SSL証明書エラー回避（推論時のモデル構築でダウンロードが発生する場合に備えて）
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # ファイル名マッチング用の正規表現 (学習時と同じ)
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
            key = get_clean_stem(img)
            if key in masks:
                self.pairs.append((img, masks[key]))

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


def build_model(model_name, encoder_name, encoder_weights, decoder_attention_type=None, encoder_depth=4):
    # HRNetなどの一部のモデルは depth=4 で学習されている場合があるため調整
    # 引数で指定された場合はそれを優先するが、HRNetの場合は4が推奨されることが多い
    # 学習スクリプトに合わせて、引数の encoder_depth をそのまま使用する形にします
    
    params = dict(encoder_name=encoder_name, encoder_weights=encoder_weights or None,
                  in_channels=3, classes=1, encoder_depth=encoder_depth)
    
    # 【追加】デコーダーのアテンションタイプを指定 (scseなど)
    if decoder_attention_type is not None:
        params["decoder_attention_type"] = decoder_attention_type

    if encoder_depth == 3:
        # 【追加】depth=3 の場合のデコーダーチャンネル設定
        params["decoder_channels"] = (256, 128, 64)
    elif encoder_depth == 4:
        # depth=4 の場合のデコーダーチャンネル設定
        params["decoder_channels"] = (256, 128, 64, 32)
    elif encoder_depth == 5:
        # depth=5 の場合のデコーダーチャンネル設定 (明示的に指定)
        params["decoder_channels"] = (256, 128, 64, 32, 16)

    # 【修正】引数の model_name を尊重するように戻す
    if model_name == "unet":
        model = smp.Unet(**params)
    elif model_name == "unet++":
        model = smp.UnetPlusPlus(**params)
    else:
        # attention_unet は元々 attention を持っているが、追加で指定も可能
        model = smp.Unet(decoder_attention_type="scse", **params)
    return model.to(DEVICE)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict and evaluate Unet soft masks")
    parser.add_argument("--data-dir", type=str, default="Kuzushiji_Restoration/datasets/dataset_final_hiragana")
    
    # 【変更】指定されたベースラインモデルの重み
    parser.add_argument("--weights", type=str, default="Kuzushiji_Restoration/experiments/segmentation/baseline/unet_baseline.pth")
    
    # 【変更】学習スクリプトに合わせてモデルを unet に変更
    parser.add_argument("--model", choices=["unet", "unet++", "attention_unet"], default="unet")
    
    # エンコーダー (se_resnext50_32x4d)
    parser.add_argument("--encoder-name", type=str, default="se_resnext50_32x4d")
    
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    
    # 【変更】出力先ディレクトリを baseline 用に変更
    parser.add_argument("--output-dir", type=str, default="Kuzushiji_Restoration/experiments/segmentation/baseline/unet")
    
    parser.add_argument("--wandb-project", type=str, default="Kuzushiji_Restoration")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    # 【変更】ベースラインは scSE なし (デフォルトFalse)
    parser.add_argument("--use-scse", action="store_true", help="Use scSE attention in decoder")

    # 【変更】ベースラインは Depth 5
    parser.add_argument("--encoder-depth", type=int, default=5, help="Encoder depth (default: 5)")

    # 【変更】ベースラインは Head Dropout なし
    parser.add_argument("--head-dropout", type=float, default=0.0, help="Dropout rate before segmentation head")

    # 【変更】ベースラインは Half Channels なし (デフォルトFalse)
    parser.add_argument("--half-decoder-channels", action="store_true", help="Use half channels in decoder (for baseline)")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    use_wandb = not args.no_wandb
    scse_tag = "_scSE" if args.use_scse else ""
    run_name = f"PredAll_{format_name(args.model)}_{format_encoder_name(args.encoder_name)}{scse_tag}"
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args))
        columns = ["id", "image", "ground_truth", "prediction"]
        test_table = wandb.Table(columns=columns)

    decoder_attention_type = 'scse' if args.use_scse else None
    model = build_model(args.model, args.encoder_name, args.encoder_weights, decoder_attention_type, args.encoder_depth)
    if args.head_dropout > 0:
        import torch.nn as nn
        logging.info(f"Inserting Dropout2d(p={args.head_dropout}) before segmentation head to match state_dict.")
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=args.head_dropout),
            model.segmentation_head
        )

    logging.info(f"Loading weights from {args.weights}")
    state = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    transform = build_transforms(args.image_size)
    subsets = ["test"]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for subset in subsets:
        logging.info(f"Processing subset: {subset}")
        img_dir = os.path.join(args.data_dir, "lq5", subset)
        mask_dir = os.path.join(args.data_dir, "mask_gt", subset)
        out_dir = os.path.join(args.output_dir, subset)
        os.makedirs(out_dir, exist_ok=True)

        try:
            dataset = EvalDataset(img_dir, mask_dir, transform)
        except FileNotFoundError:
            logging.warning(f"Skipping {subset}: Dataset not found in {img_dir} or {mask_dir}")
            continue

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        mae_list, soft_iou_list, hard_iou_list, f1_list = [], [], [], []

        with torch.no_grad():
            for names, imgs, masks in tqdm(loader, desc=f"Predict {subset}"):
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
                    Image.fromarray(mask_img).save(os.path.join(out_dir, Path(name).with_suffix(".jpg")))

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
        logging.info(f"Results for {subset}: {metrics}")
        if use_wandb:
            wandb.log(metrics)
            wandb.log({"evaluation_samples": test_table})
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()