#!/usr/bin/env python3
import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms as T

# try segmentation_models_pytorch DeepLabV3+ first
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False

# fallback to torchvision deeplabv3
try:
    import torchvision.models.segmentation as seg_models
    TV_AVAILABLE = True
except Exception:
    TV_AVAILABLE = False

def makedirs(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def list_images(indir):
    exts = ('*.png','*.jpg','*.jpeg','*.tif','*.tiff')
    files = []
    for e in exts:
        files.extend(glob(os.path.join(indir,e)))
    return sorted(files)

def load_image(path, resize=None):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"failed load {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    orig_shape = im.shape[:2]  # H,W
    if resize:
        im = cv2.resize(im, (resize, resize), interpolation=cv2.INTER_AREA)
    return im, orig_shape

def postprocess_mask(prob, thresh=0.5, morph=False, min_area=0):
    # prob: HxW float [0,1]
    binm = (prob >= thresh).astype('uint8') * 255
    if morph:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k)
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    if min_area > 0:
        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_clean = np.zeros_like(binm)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(mask_clean, [c], -1, 255, thickness=cv2.FILLED)
        binm = mask_clean
    return binm

def build_model(num_classes=1, device='cpu', use_deeplabv3plus=True):
    if use_deeplabv3plus and SMP_AVAILABLE:
        model = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', classes=num_classes, activation=None)
    elif TV_AVAILABLE:
        # torchvision DeeplabV3 (fallback). num_classes sets final classifier channels.
        model = seg_models.deeplabv3_resnet50(pretrained_backbone=True, progress=True, num_classes=num_classes)
    else:
        raise RuntimeError("No segmentation model available: install segmentation_models_pytorch or torchvision")
    return model.to(device)

def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path:
        return
    ck = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ck, dict) and ('model_state_dict' in ck or 'state_dict' in ck):
        sd = ck.get('model_state_dict', ck.get('state_dict', ck))
        model.load_state_dict(sd, strict=False)
    elif isinstance(ck, dict):
        # try to use as state dict
        if 'state_dict' in ck:
            model.load_state_dict(ck['state_dict'], strict=False)
        else:
            # may already be state_dict-like
            model.load_state_dict(ck, strict=False)
    else:
        model.load_state_dict(ck, strict=False)
    model.to(device)

def predict_batch(model, batch_tensor, device):
    model.eval()
    with torch.no_grad():
        batch_tensor = batch_tensor.to(device)
        out = model(batch_tensor)
        if isinstance(out, dict) and 'out' in out:
            logits = out['out']
        else:
            logits = out
        # logits shape: N x C x H x W
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:,0:1,:,:])
        else:
            # if multi-class, take foreground as non-zero classes merged or argmax >0
            probs = torch.softmax(logits, dim=1)
            # merge classes >0 into foreground prob = max_{cls>0} prob
            fg = probs[:,1:,...].max(dim=1, keepdim=True)[0] if probs.shape[1] > 1 else probs[:,0:1,...]
            probs = fg
        probs_cpu = probs.cpu().numpy()  # N x 1 x H x W
        return probs_cpu[:,0,:,:]

def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ predict masks (binary) for text")
    parser.add_argument('--in-dir', required=True, help='input images directory')
    parser.add_argument('--out-dir', default='results/deeplab_pred', help='output masks dir')
    parser.add_argument('--checkpoint', default=None, help='model checkpoint (state_dict or ckpt)')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resize', type=int, default=512, help='resize short side (square) for model input')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--morph', action='store_true', help='apply open+close morphological filter')
    parser.add_argument('--min-area', type=int, default=30, help='remove small components under this area (px)')
    parser.add_argument('--save-prob', action='store_true', help='save probability maps as .npy alongside masks')
    parser.add_argument('--num-classes', type=int, default=1, help='model output channels (1 = binary prob, >1 = multiclass)')
    args = parser.parse_args()

    makedirs(args.out_dir)
    img_paths = list_images(args.in_dir)
    if len(img_paths) == 0:
        print("no images found in", args.in_dir); return

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    use_deeplabv3plus = True
    if not SMP_AVAILABLE and not TV_AVAILABLE:
        raise RuntimeError("Neither segmentation_models_pytorch nor torchvision available in environment")

    model = build_model(num_classes=args.num_classes, device=device, use_deeplabv3plus=use_deeplabv3plus)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)

    # transforms
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((args.resize, args.resize)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # batching
    batch = []
    names = []
    orig_shapes = []
    for p in tqdm(img_paths, desc="Predicting"):
        try:
            im, orig_shape = load_image(p, resize=args.resize)
        except RuntimeError:
            print("skip", p); continue
        inp = tf(im)  # C,H,W
        batch.append(inp)
        names.append(os.path.basename(p))
        orig_shapes.append(orig_shape)
        if len(batch) >= args.batch_size:
            bt = torch.stack(batch, dim=0)
            probs_batch = predict_batch(model, bt, device)  # N x H x W
            for name, prob_map, orig_shape in zip(names, probs_batch, orig_shapes):
                # resize prob back to original
                prob_resized = cv2.resize((prob_map*255).astype('uint8'), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR) / 255.0
                mask = postprocess_mask(prob_resized, thresh=args.threshold, morph=args.morph, min_area=args.min_area)
                out_mask_p = os.path.join(args.out_dir, os.path.splitext(name)[0] + "_prediction.png")
                cv2.imwrite(out_mask_p, mask)
                if args.save_prob:
                    np.save(os.path.splitext(out_mask_p)[0] + ".npy", prob_resized.astype(np.float32))
            batch = []; names = []; orig_shapes = []

    # final partial batch
    if batch:
        bt = torch.stack(batch, dim=0)
        probs_batch = predict_batch(model, bt, device)
        for name, prob_map, orig_shape in zip(names, probs_batch, orig_shapes):
            prob_resized = cv2.resize((prob_map*255).astype('uint8'), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR) / 255.0
            mask = postprocess_mask(prob_resized, thresh=args.threshold, morph=args.morph, min_area=args.min_area)
            out_mask_p = os.path.join(args.out_dir, os.path.splitext(name)[0] + "_prediction.png")
            cv2.imwrite(out_mask_p, mask)
            if args.save_prob:
                np.save(os.path.splitext(out_mask_p)[0] + ".npy", prob_resized.astype(np.float32))

    print("Saved masks to", args.out_dir)

if __name__ == '__main__':
    main()