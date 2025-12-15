#!/usr/bin/env python3
import os
import logging
import argparse
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.image_dir}")
        self.paths = sorted([p for p in self.image_dir.iterdir() if p.is_file()])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return path.name, img

def build_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def build_model(encoder, encoder_weights):
    enc_weights = None if str(encoder_weights).lower() in ("none", "null") else encoder_weights
    return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=enc_weights,
                             in_channels=3, classes=1).to(DEVICE)

def parse_args():
    parser = argparse.ArgumentParser(description="DeepLabV3+ soft mask prediction")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", default="efficientnet-b4")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Predicting soft masks: %s -> %s", args.input_dir, args.output_dir)

    transforms = build_transforms(args.image_size)
    dataset = InferenceDataset(args.input_dir, transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.encoder, args.encoder_weights)
    state = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for names, imgs in loader:
            imgs = imgs.to(DEVICE, dtype=torch.float)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            for name, prob in zip(names, probs):
                prob = prob[0]
                mask = (prob * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(mask).save(Path(args.output_dir) / Path(name).with_suffix(".png"))

    logging.info("Finished writing soft masks.")

if __name__ == "__main__":
    main()