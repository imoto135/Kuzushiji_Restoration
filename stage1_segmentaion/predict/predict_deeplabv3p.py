import os
import argparse
import logging
from PIL import Image
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_map(d, allowed_exts={'.jpg', '.jpeg', '.png'}, pref_order=['.jpg', '.jpeg', '.png']):
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

def build_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def main():
    parser = argparse.ArgumentParser(description="Predict binary text masks using DeepLabV3+")
    parser.add_argument('--model-path', type=str, required=True, help='trained DeeplabV3+ weights (.pth)')
    parser.add_argument('--data-dir', type=str, default='dataset', help='dataset root (lq and mask subfolders)')
    parser.add_argument('--lq-subdir', type=str, default='lq/test', help='low-quality images subfolder under data-dir')
    parser.add_argument('--mask-subdir', type=str, default='mask_gt/test', help='ground-truth masks subfolder (optional, used for comparison)')
    parser.add_argument('--encoder', type=str, default='efficientnet-b7', help='encoder name used at training')
    parser.add_argument('--encoder-weights', type=str, default='None', help='encoder weights used at training (imagenet or None)')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--out-dir', type=str, default='results/deeplabv3p_preds_noembed', help='output directory')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (auto if not set)')
    parser.add_argument('--conv-stem-adapt', type=str, default='mean', choices=['slice','mean','repeat'],
                    help='How to adapt checkpoint conv stem when checkpoint has more input channels than model: \n'
                        'slice = take first N channels\n'
                        'mean = average groups to reduce channels (default)\n'
                        'repeat = repeat existing channels to expand')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    lq_dir = os.path.join(args.data_dir, args.lq_subdir)
    mask_dir = os.path.join(args.data_dir, args.mask_subdir)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'pred_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'comparisons'), exist_ok=True)

    # build file maps (prefer jpg over png)
    img_map = build_map(lq_dir)
    mask_map = build_map(mask_dir)

    common_stems = sorted(img_map.keys())  # predict for all available inputs
    if len(common_stems) == 0:
        print(f"Error: 入力ディレクトリに画像が見つかりません: {lq_dir}")
        return

    transform = build_transforms(args.image_size)

    # load model
    enc_w = None if args.encoder_weights in ('None', 'none', 'NoneType') else args.encoder_weights
    model = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights=enc_w, in_channels=3, classes=1)

    # --- Robust state_dict load: handle 'module.' prefix and conv_stem channel mismatch ---
    state = torch.load(args.model_path, map_location='cpu')
    sd = state.get('state_dict', state) if isinstance(state, dict) else state

    # remove 'module.' prefix if present
    sd = { (k.replace('module.', '') if k.startswith('module.') else k): v for k, v in sd.items() }

    model_sd = model.state_dict()

    # find conv stem key in checkpoint that corresponds to model
    conv_keys = [k for k in sd.keys() if 'conv_stem' in k and 'weight' in k]
    # fallback keys to consider if different naming
    if len(conv_keys) == 0:
        conv_keys = [k for k in sd.keys() if ('conv1' in k or 'stem' in k) and 'weight' in k]

    # adjust conv stem weights if channel mismatch
    for ck in conv_keys:
        if ck in model_sd:
            w_ck = sd[ck]
            w_model = model_sd[ck]
            if w_ck.shape != w_model.shape:
                import logging
                logging.warning(f"conv stem mismatch: checkpoint {w_ck.shape} vs model {w_model.shape} for key {ck}")

                # shapes: (out_channels, in_channels, k, k)
                ck_out, ck_in = w_ck.shape[0], w_ck.shape[1]
                model_out, model_in = w_model.shape[0], w_model.shape[1]

                # First adapt input channels (second dim) if needed
                w_new = w_ck
                if ck_in != model_in:
                    mode = args.conv_stem_adapt
                    if ck_in > model_in:
                        if mode == 'slice':
                            w_new = w_new[:, :model_in, :, :].clone()
                            logging.warning(f"Sliced checkpoint conv in-channels {ck_in} -> {model_in}")
                        elif mode == 'mean':
                            base = ck_in // model_in
                            rem = ck_in % model_in
                            idx = 0
                            parts = []
                            for i in range(model_in):
                                sz = base + (1 if i < rem else 0)
                                parts.append(w_new[:, idx:idx+sz, :, :].mean(dim=1, keepdim=True))
                                idx += sz
                            w_new = torch.cat(parts, dim=1)
                            logging.warning(f"Averaged checkpoint conv in-channels {ck_in} -> {model_in} (mean)")
                        else:  # repeat
                            w_new = w_new[:, :model_in, :, :].clone()
                            logging.warning(f"Sliced (repeat-mode fallback) checkpoint conv in-channels {ck_in} -> {model_in}")
                    else:
                        # expand by repeating along in-channel dim
                        reps = model_in // ck_in
                        rem = model_in - reps * ck_in
                        w_new = w_new.repeat(1, reps, 1, 1)
                        if rem > 0:
                            w_new = torch.cat([w_new, w_ck[:, :rem, :, :]], dim=1)
                        w_new = w_new[:, :model_in, :, :].clone()
                        logging.warning(f"Padded checkpoint conv in-channels {ck_in} -> {model_in}")

                # Now adapt output channels (first dim) if needed
                ck_out_new = w_new.shape[0]
                if ck_out_new != model_out:
                    mode = args.conv_stem_adapt
                    if ck_out_new > model_out:
                        if mode == 'slice':
                            w_new = w_new[:model_out, :, :, :].clone()
                            logging.warning(f"Sliced checkpoint conv out-channels {ck_out_new} -> {model_out}")
                        elif mode == 'mean':
                            base = ck_out_new // model_out
                            rem = ck_out_new % model_out
                            idx = 0
                            parts = []
                            for i in range(model_out):
                                sz = base + (1 if i < rem else 0)
                                parts.append(w_new[idx:idx+sz, :, :, :].mean(dim=0, keepdim=True))
                                idx += sz
                            w_new = torch.cat(parts, dim=0)
                            logging.warning(f"Averaged checkpoint conv out-channels {ck_out_new} -> {model_out} (mean)")
                        else:  # repeat fallback
                            w_new = w_new[:model_out, :, :, :].clone()
                            logging.warning(f"Sliced (repeat-mode fallback) checkpoint conv out-channels {ck_out_new} -> {model_out}")
                    else:
                        # expand by repeating output filters
                        reps = model_out // ck_out_new
                        rem = model_out - reps * ck_out_new
                        w_new = w_new.repeat(reps, 1, 1, 1)
                        if rem > 0:
                            w_new = torch.cat([w_new, w_new[:rem, :, :, :]], dim=0)
                        w_new = w_new[:model_out, :, :, :].clone()
                        logging.warning(f"Padded checkpoint conv out-channels {ck_out_new} -> {model_out}")

                # assign adapted weight back to sd
                sd[ck] = w_new

    # drop other keys whose shapes are totally incompatible to avoid load errors
    keys_to_del = []
    for k, v in sd.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            # すでに handled conv stem; それ以外は削除して model 初期化値を使う
            if 'conv_stem' not in k and 'conv1' not in k and 'stem' not in k:
                keys_to_del.append(k)
    for k in keys_to_del:
        sd.pop(k, None)
        logging.warning(f"Dropped incompatible key from checkpoint: {k}")

    # finally load with strict=False
    model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()

    for stem in common_stems:
        try:
            img_fname = img_map[stem]
            img_path = os.path.join(lq_dir, img_fname)
            pil = Image.open(img_path).convert("RGB")
            img_np = np.array(pil)

            aug = transform(image=img_np)
            input_tensor = aug['image'].unsqueeze(0).to(device, dtype=torch.float)

            with torch.no_grad():
                out = model(input_tensor)
                prob = torch.sigmoid(out)
                pred = (prob > args.threshold).float().squeeze(0).squeeze(0).cpu().numpy()  # HxW

            pred_uint8 = (pred * 255).astype(np.uint8)
            pred_path = os.path.join(args.out_dir, 'pred_masks', f"{stem}.png")
            cv2.imwrite(pred_path, pred_uint8)

            # 比較画像（元画像 | GTマスク(if exists) | 予測マスク）
            orig_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            h, w = orig_bgr.shape[:2]

            gt_exists = stem in mask_map
            if gt_exists:
                gt_bgr = cv2.imread(os.path.join(mask_dir, mask_map[stem]))
                if gt_bgr is None:
                    gt_bgr = np.zeros_like(orig_bgr)
                else:
                    gt_bgr = cv2.resize(gt_bgr, (w, h))
            else:
                gt_bgr = np.zeros_like(orig_bgr)

            pred_bgr = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2BGR)
            pred_bgr = cv2.resize(pred_bgr, (w, h))

            comparison = np.concatenate([orig_bgr, gt_bgr, pred_bgr], axis=1)
            comp_path = os.path.join(args.out_dir, 'comparisons', f"{stem}.png")
            cv2.imwrite(comp_path, comparison)

        except Exception as e:
            print(f"Failed {stem}: {e}")

    print(f"Predictions saved to {args.out_dir}")

if __name__ == "__main__":
    main()