#!/usr/bin/env python3
"""
Run DeeplabV3/DeepLabV3+ model checkpoint to predict per-pixel class ids
Saves single-channel PNGs where pixel value == class_id (uint8).

Default checkpoint: embedding_deeplabv3p_best_model.pth (repo root)
Default output dir: results/deeplabv3p_mask

Forward interface follows scripts/train_deeplab.py: model(inputs)['out'] -> logits (N,C,H,W)
"""
import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torchvision import transforms as T

# try segmentation_models_pytorch DeepLabV3+ first
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False

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

def build_model(num_classes=1, device='cpu', use_deeplabv3plus=True, encoder_name='resnet50', in_channels=3):
    if use_deeplabv3plus and SMP_AVAILABLE:
        # segmentation_models_pytorch supports specifying encoder_name and in_channels
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights='imagenet', in_channels=in_channels, classes=num_classes, activation=None)
    elif TV_AVAILABLE:
        # torchvision models don't support custom in_channels easily; raise informative error
        if in_channels != 3:
            raise RuntimeError('torchvision DeepLabV3 does not support custom in_channels; install segmentation_models_pytorch')
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
        if 'state_dict' in ck:
            model.load_state_dict(ck['state_dict'], strict=False)
        else:
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
        # logits: N x C x H x W
        probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
        # for class-id output, take argmax over channels
        if logits.shape[1] > 1:
            preds = logits.argmax(dim=1).cpu().numpy()  # N x H x W
        else:
            # single-channel logits -> binary: 0 or 1
            p = torch.sigmoid(logits[:,0:1,...])
            preds = (p[:,0,...] >= 0.5).cpu().numpy().astype(np.uint8)  # N x H x W
        return preds

def main():
    parser = argparse.ArgumentParser(description='DeeplabV3p per-pixel class prediction (restore masks)')
    parser.add_argument('--in-dir', required=True, help='input (damaged) images directory')
    parser.add_argument('--out-dir', default='results/deeplabv3p_mask', help='output masks dir')
    parser.add_argument('--checkpoint', default='embedding_deeplabv3p_best_model.pth', help='checkpoint path')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resize', type=int, default=128, help='square resize for model input')
    parser.add_argument('--num-classes', type=int, default=1, help='model output channels (1 = binary mask)')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary mask when num-classes==1')
    parser.add_argument('--save-prob', action='store_true', help='save probability maps (.npy) alongside mask PNGs')
    parser.add_argument('--debug', action='store_true', help='print per-image probability/logit stats for debugging')
    parser.add_argument('--encoder', default='efficientnet-b7', help='encoder name for smp DeepLabV3Plus (e.g. efficientnet-b7)')
    parser.add_argument('--in-channels', type=int, default=67, help='number of input channels for model (e.g. 3 + EMBEDDING_DIM)')
    args = parser.parse_args()

    makedirs(args.out_dir)
    img_paths = list_images(args.in_dir)
    if len(img_paths) == 0:
        print('no images found in', args.in_dir); return

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # determine num_classes: user-specified (default 1 for this checkpoint)
    num_classes = args.num_classes

    if not SMP_AVAILABLE and not TV_AVAILABLE:
        raise RuntimeError('Neither segmentation_models_pytorch nor torchvision available in environment')

    model = build_model(num_classes=num_classes, device=device, use_deeplabv3plus=True, encoder_name=args.encoder, in_channels=args.in_channels)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((args.resize, args.resize)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    batch = []
    names = []
    orig_shapes = []
    for p in tqdm(img_paths, desc='Predicting'):
        try:
            im, orig_shape = load_image(p, resize=args.resize)
        except RuntimeError:
            print('skip', p); continue
        inp = tf(im)
        # tf gives C,H,W tensor; if model expects extra channels (embedding map), append zeros
        import torch as _torch
        if args.in_channels > 3:
            C, H, W = inp.shape
            extra_ch = args.in_channels - 3
            extra = _torch.zeros((extra_ch, H, W), dtype=inp.dtype)
            inp = _torch.cat([inp, extra], dim=0)
        batch.append(inp)
        names.append(os.path.basename(p))
        orig_shapes.append(orig_shape)
        if len(batch) >= args.batch_size:
            bt = torch.stack(batch, dim=0)
            # get raw model outputs and optionally probability maps
            model.eval()
            with torch.no_grad():
                out = model(bt.to(device))
                logits = out['out'] if isinstance(out, dict) and 'out' in out else out
                # logits shape: N x C x H x W
                if logits.shape[1] == 1:
                    probs = torch.sigmoid(logits)[:,0].cpu().numpy()  # N x H x W
                    preds_batch = (probs >= args.threshold).astype('uint8')
                else:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds_batch = logits.argmax(dim=1).cpu().numpy()
            for i, (name, orig_shape) in enumerate(zip(names, orig_shapes)):
                pred_map = preds_batch[i]
                # if binary, pred_map is 0/1; convert to uint8
                if logits.shape[1] == 1:
                    prob_map = probs[i]
                    if args.debug:
                        print(f"{name}: prob min={prob_map.min():.4f} mean={prob_map.mean():.4f} max={prob_map.max():.4f}")
                    pred_resized = cv2.resize((pred_map*255).astype('uint8'), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                    out_p = os.path.join(args.out_dir, os.path.splitext(name)[0] + '.png')
                    cv2.imwrite(out_p, pred_resized)
                    if args.save_prob:
                        prob_resized = cv2.resize((prob_map*255).astype('uint8'), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR) / 255.0
                        np.save(os.path.splitext(out_p)[0] + '.npy', prob_resized.astype(np.float32))
                else:
                    # multiclass: pred_map contains class ids
                    pred_resized = cv2.resize(pred_map.astype('uint8'), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                    out_p = os.path.join(args.out_dir, os.path.splitext(name)[0] + '.png')
                    cv2.imwrite(out_p, pred_resized)
            batch = []; names = []; orig_shapes = []

    if batch:
        bt = torch.stack(batch, dim=0)
        model.eval()
        with torch.no_grad():
            out = model(bt.to(device))
            logits = out['out'] if isinstance(out, dict) and 'out' in out else out
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)[:,0].cpu().numpy()
                preds_batch = (probs >= args.threshold).astype('uint8')
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds_batch = logits.argmax(dim=1).cpu().numpy()
        for i, (name, orig_shape) in enumerate(zip(names, orig_shapes)):
            pred_map = preds_batch[i]
            if logits.shape[1] == 1:
                prob_map = probs[i]
                if args.debug:
                    print(f"{name}: prob min={prob_map.min():.4f} mean={prob_map.mean():.4f} max={prob_map.max():.4f}")
                pred_resized = cv2.resize((pred_map*255).astype('uint8'), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                out_p = os.path.join(args.out_dir, os.path.splitext(name)[0] + '.png')
                cv2.imwrite(out_p, pred_resized)
                if args.save_prob:
                    prob_resized = cv2.resize((prob_map*255).astype('uint8'), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR) / 255.0
                    np.save(os.path.splitext(out_p)[0] + '.npy', prob_resized.astype(np.float32))
            else:
                pred_resized = cv2.resize(pred_map.astype('uint8'), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                out_p = os.path.join(args.out_dir, os.path.splitext(name)[0] + '.png')
                cv2.imwrite(out_p, pred_resized)

    print('Saved masks to', args.out_dir)

if __name__ == '__main__':
    main()
