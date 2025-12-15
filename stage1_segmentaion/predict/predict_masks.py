#!/usr/bin/env python3
import os
import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = Path(image_dir)
        self.paths = sorted([p for p in self.image_dir.iterdir() if p.is_file()])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)['image']
        return path.name, img

def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def build_model(model_name, encoder_weights, device):
    params = dict(encoder_name="efficientnet-b7", encoder_weights=encoder_weights or None,
                  in_channels=3, classes=1)
    if model_name == "unet":
        model = smp.Unet(**params)
    elif model_name == "unet++":
        model = smp.UnetPlusPlus(**params)
    else:
        model = smp.Unet(decoder_attention_type="scse", **params)
    return model.to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Predict soft masks")
    parser.add_argument("--model", choices=["unet","unet++","attention_unet"], default="unet++")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--encoder-weights", default="imagenet")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Predicting soft masks with %s", args.weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(args.image_size)
    dataset = InferenceDataset(args.input_dir, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, args.encoder_weights, device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for names, imgs in loader:
            imgs = imgs.to(device, dtype=torch.float)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            for name, prob in zip(names, probs):
                prob = prob[0]
                mask = (prob * 255.0).clip(0, 255).astype(np.uint8)
                out_path = Path(args.output_dir) / Path(name).with_suffix(".jpg")
                Image.fromarray(mask).convert("L").save(out_path, format="JPEG", quality=95)

    logging.info("Saved predictions to %s", args.output_dir)

if __name__ == "__main__":
    main()