#!/usr/bin/env python3
"""
DeeplabV3 を使ってバイナリ文字マスクを推論するスクリプト。

使い方の例:
  conda run -n restormer_environment python scripts/deeplab_predict_masks.py \
    --in-dir results/samples --out-dir results/samples_deeplab_pred --checkpoint path/to.ckpt \
    --device cuda --batch-size 4 --resize 512 --threshold 0.5 --morph --min-area 30

このスクリプトは既存の `unet++_conditional.py` の学習設定や出力命名ルールを変えず、
推論専用のツールとして追加しています。
"""

import os
import sys
import argparse
import csv
from glob import glob
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms


def get_image_paths(in_dir, exts=(".png", ".jpg", ".jpeg")):
    p = Path(in_dir)
    files = []
    for e in exts:
        files.extend(sorted(p.rglob(f"*{e}")))
    return [str(x) for x in files]


def build_model(num_classes=1, device="cpu"):
    # Binary segmentation: use num_classes=1 and sigmoid output
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    model.to(device)
    model.eval()
    return model


def preprocess_img(img_bgr, resize=None):
    # img_bgr: HxWx3 BGR uint8
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    # ImageNet normalization (common for pretrained backbones)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    return img


def postprocess_and_save(prob_map, out_path, orig_shape=None, threshold=0.5, morph=False, min_area=0):
    # prob_map: numpy HxW (float 0..1)
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    if orig_shape is not None and mask.shape != (orig_shape[0], orig_shape[1]):
        mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    if morph:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

    if min_area > 0:
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_mask = np.zeros_like(mask)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(out_mask, [c], -1, 255, thickness=cv2.FILLED)
        mask = out_mask

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mask)


def load_checkpoint(model, ckpt_path, device="cpu"):
    if not ckpt_path:
        return model
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)
    # allow both raw state_dict and {'state_dict':...}
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # fix keys if they were saved with 'module.' prefix
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', required=True, help='入力画像ディレクトリ（推論対象）')
    parser.add_argument('--out-dir', required=True, help='出力マスク保存ディレクトリ')
    parser.add_argument('--checkpoint', default=None, help='モデルのチェックポイント（torch）')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resize', type=int, default=None, help='モデル入力にリサイズするサイズ（正方形）')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--morph', action='store_true', help='開閉処理を行う')
    parser.add_argument('--min-area', type=int, default=0, help='最小領域フィルタ（ピクセル）')
    parser.add_argument('--save-prob', action='store_true', help='確率マップ（float）も保存する')
    parser.add_argument('--exts', nargs='+', default=['.png', '.jpg', '.jpeg'])
    parser.add_argument('--class-map', default=None, help='class_map.csv のパス（char_unicode,class_id の列を想定）')
    parser.add_argument('--class-thresh-csv', default=None, help='クラスごとの後処理パラメータ CSV（class_id,threshold,morph,min_area）')
    args = parser.parse_args()

    img_paths = get_image_paths(args.in_dir, exts=tuple(args.exts))
    if len(img_paths) == 0:
        print('No images found in', args.in_dir)
        sys.exit(1)

    device = torch.device(args.device)
    model = build_model(num_classes=1, device=device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device=str(device))

    # --- class map / per-class params ---
    char_to_id = {}
    if args.class_map:
        if os.path.exists(args.class_map):
            try:
                with open(args.class_map, newline='') as f:
                    reader = csv.DictReader(f)
                    # try common column names
                    for r in reader:
                        if 'char_unicode' in r and 'class_id' in r:
                            char_to_id[r['char_unicode']] = int(r['class_id'])
                        else:
                            # fallback to first two columns
                            keys = list(r.keys())
                            char_to_id[r[keys[0]]] = int(r[keys[1]])
                print(f"Loaded class_map with {len(char_to_id)} entries from {args.class_map}")
            except Exception as e:
                print('Failed to read class_map:', e)
        else:
            print('class_map path does not exist:', args.class_map)

    # per-class params: dict[class_id] = {'threshold':..., 'morph':bool, 'min_area':int}
    class_params = {}
    if args.class_thresh_csv:
        if os.path.exists(args.class_thresh_csv):
            try:
                with open(args.class_thresh_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        cid = int(r.get('class_id') or r.get('class') or list(r.values())[0])
                        thresh = float(r.get('threshold') or r.get('thresh') or args.threshold)
                        morph = str(r.get('morph') or r.get('do_morph') or 'False').lower() in ('1','true','yes')
                        min_area = int(r.get('min_area') or r.get('minarea') or 0)
                        class_params[cid] = {'threshold':thresh, 'morph':morph, 'min_area':min_area}
                print(f"Loaded per-class params for {len(class_params)} classes from {args.class_thresh_csv}")
            except Exception as e:
                print('Failed to read class-thresh-csv:', e)
        else:
            print('class-thresh-csv path does not exist:', args.class_thresh_csv)

    transform = None

    bs = max(1, args.batch_size)
    # process in simple batches
    for i in tqdm(range(0, len(img_paths), bs), desc='Predicting'):
        batch_paths = img_paths[i:i+bs]
        imgs = []
        orig_shapes = []
        for p in batch_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                print('Failed to read', p)
                img = np.zeros((args.resize or 256, args.resize or 256, 3), dtype=np.uint8)
            orig_shapes.append(img.shape[:2])
            arr = preprocess_img(img, resize=args.resize)
            imgs.append(arr)

        x = np.stack(imgs, axis=0)
        xt = torch.from_numpy(x).to(device)

        with torch.no_grad():
            out = model(xt)
            # model returns dict with 'out'
            if isinstance(out, dict) and 'out' in out:
                logits = out['out']
            else:
                logits = out
            # logits: NxCxhxw
            # for binary (C==1) use sigmoid
            logits = logits.cpu()
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits[:, 0, :, :]).numpy()
            else:
                # multiclass: take class 1 probability via softmax
                probs = torch.softmax(logits, dim=1)[:, 1, :, :].numpy()

        # save each
        for j, p in enumerate(batch_paths):
            prob_map = probs[j]
            stem = Path(p).stem
            out_name = stem + '_prediction.png'
            out_path = os.path.join(args.out_dir, out_name)

            # determine class id from filename prefix using class_map if available
            # filename pattern expected like 'U+3042_...'
            filename = Path(p).name
            prefix = filename.split('_')[0]
            class_id = None
            if prefix in char_to_id:
                class_id = char_to_id[prefix]

            # if per-class params provided, override defaults
            if class_id is not None and class_id in class_params:
                cp = class_params[class_id]
                thresh = cp.get('threshold', args.threshold)
                morph = cp.get('morph', args.morph)
                min_area = cp.get('min_area', args.min_area)
            else:
                thresh = args.threshold
                morph = args.morph
                min_area = args.min_area

            postprocess_and_save(prob_map, out_path, orig_shape=orig_shapes[j], threshold=thresh, morph=morph, min_area=min_area)
            if args.save_prob:
                prob_path = os.path.join(args.out_dir, stem + '_prob.npy')
                np.save(prob_path, prob_map)

    print('Done. Predicted masks saved to', args.out_dir)


if __name__ == '__main__':
    main()
