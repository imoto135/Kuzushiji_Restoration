#!/usr/bin/env python3
import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import pandas as pd

# --- utils ---
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
    orig_shape = im.shape[:2]
    if resize:
        im = cv2.resize(im, (resize, resize), interpolation=cv2.INTER_AREA)
    return im, orig_shape

def postprocess_mask(prob, thresh=0.5, morph=False, min_area=0):
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

# --- simple UNet fallback ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_channels, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.u2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.u1 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_channels, 1)
    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        x = self.up2(d3)
        x = self.u2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.u1(torch.cat([x, d1], dim=1))
        return self.outc(x)

# --- model loader attempt: try import from unet++_conditional.py ---
def try_import_conditional_model(in_channels, out_channels, device):
    try:
        import unet__conditional as cond_mod  # try different name
    except Exception:
        try:
            import unet_plus_plus_conditional as cond_mod
        except Exception:
            try:
                import unet__conditional as cond_mod  # fallback exact name attempt
            except Exception:
                cond_mod = None
    if cond_mod:
        for attr in ('build_model','get_model','create_model','ConditionalUNet','UNetConditional'):
            if hasattr(cond_mod, attr):
                fn = getattr(cond_mod, attr)
                try:
                    m = fn(in_channels=in_channels, num_classes=out_channels)
                    return m.to(device)
                except Exception:
                    pass
    return None

def _extract_embedding_tensor(ck):
    """
    ck: result of torch.load(...)
    戻り値: torch.Tensor の embedding weight (1D or 2D) または None
    探索順:
      - ck['emb_state_dict'] (and keys like 'weight' or '.weight')
      - ck['emb'] / ck['embedding'] / ck['embedding.weight']
      - ck['state_dict'] 内の 'emb' を含むキー
      - ck が直接 tensor の場合はそれを返す
    """
    if ck is None:
        return None
    # 直接 tensor
    if isinstance(ck, torch.Tensor):
        return ck
    # dict-like
    if isinstance(ck, dict):
        # common keys
        if 'emb_state_dict' in ck:
            sd = ck['emb_state_dict']
            if isinstance(sd, dict):
                # look for weight
                for k in ('weight','emb.weight','embedding.weight'):
                    if k in sd:
                        return sd[k]
                # fallback: first tensor value
                for v in sd.values():
                    if isinstance(v, torch.Tensor):
                        return v
        # direct weight fields in ck
        for k in ('emb.weight','embedding.weight','emb'):
            if k in ck and isinstance(ck[k], torch.Tensor):
                return ck[k]
        # top-level 'state_dict'
        if 'state_dict' in ck and isinstance(ck['state_dict'], dict):
            sd = ck['state_dict']
            # prefer keys that contain 'emb'
            for key, val in sd.items():
                if 'emb' in key and isinstance(val, torch.Tensor):
                    return val
            # fallback: first tensor
            for v in sd.values():
                if isinstance(v, torch.Tensor):
                    return v
        # fallback: any tensor value in ck
        for v in ck.values():
            if isinstance(v, torch.Tensor):
                return v
    return None

# --- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', required=True)
    parser.add_argument('--out-dir', default='results/unet_pred_masks')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--morph', action='store_true')
    parser.add_argument('--min-area', type=int, default=30)
    parser.add_argument('--save-prob', action='store_true')
    parser.add_argument('--class-map', default='class_map.csv')
    parser.add_argument('--embedding-pth', default=None, help='optional saved embedding weights (tensor num_classes x EMBEDDING_DIM)')
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--in-channels', type=int, default=None, help='override input channels; default 3+embedding_dim')
    parser.add_argument('--num-classes', type=int, default=1, help='output channels (1 = binary prob)')
    args = parser.parse_args()

    makedirs(args.out_dir)
    img_paths = list_images(args.in_dir)
    if len(img_paths)==0:
        print("no images in", args.in_dir); return

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # load class_map
    class_map = {}
    if os.path.isfile(args.class_map):
        try:
            df = pd.read_csv(args.class_map)
            class_map = pd.Series(df.class_id.values, index=df.char_unicode).to_dict()
        except Exception:
            pass

    # load embedding weights if provided
    embedding_weights = None
    if args.embedding_pth and os.path.isfile(args.embedding_pth):
        try:
            raw = torch.load(args.embedding_pth, map_location='cpu')
        except Exception as e:
            print("Failed to load embedding file:", args.embedding_pth, "error:", e)
            raise

        embedding_tensor = _extract_embedding_tensor(raw)
        if embedding_tensor is None:
            raise RuntimeError(f"Could not find embedding tensor in {args.embedding_pth}; keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}")

        # ensure tensor
        if not isinstance(embedding_tensor, torch.Tensor):
            embedding_tensor = torch.tensor(embedding_tensor)

        embedding_weights = embedding_tensor.clone().cpu()

    # --- 追加: チェックポイント内に埋め込みが含まれている場合はそれを使う ---
    if embedding_weights is None and args.checkpoint and os.path.isfile(args.checkpoint):
        try:
            ck_for_emb = torch.load(args.checkpoint, map_location='cpu')
            emb_from_ck = _extract_embedding_tensor(ck_for_emb)
            if emb_from_ck is not None:
                if not isinstance(emb_from_ck, torch.Tensor):
                    emb_from_ck = torch.tensor(emb_from_ck)
                embedding_weights = emb_from_ck.clone().cpu()
                # update embedding-dim (args.embedding_dim) so IN_CH 保持が正しくなる
                if embedding_weights.ndim == 2:
                    args.embedding_dim = embedding_weights.shape[1]
                else:
                    args.embedding_dim = embedding_weights.shape[0]
                print(f"Extracted embedding (dim={args.embedding_dim}) from checkpoint: {args.checkpoint}")
        except Exception as e:
            print("Warning: failed to extract embedding from checkpoint:", e)
    # --- ここまで追加 ---

    EMB_DIM = args.embedding_dim
    IN_CH = args.in_channels if args.in_channels else 3 + EMB_DIM

    # try to import conditional model from repo
    model = try_import_conditional_model(IN_CH, args.num_classes, device)
    if model is None:
        model = SimpleUNet(in_channels=IN_CH, out_channels=args.num_classes)
        model.to(device)

    # load checkpoint state dict flexibly
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location='cpu')
        sd = None
        if isinstance(ck, dict):
            for key in ('model_state_dict','state_dict','model'):
                if key in ck:
                    sd = ck[key]; break
            if sd is None:
                # maybe full state dict provided
                sd = ck
        else:
            sd = ck
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print("warning: load_state_dict failed strict=False, trying partial keys:", e)
            try:
                model.load_state_dict(sd, strict=False)
            except Exception as e2:
                print("final load failure:", e2)

    # transforms
    mean = [0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((args.resize, args.resize)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    batch_imgs = []
    batch_names = []
    batch_orig = []
    for p in tqdm(img_paths, desc="Predict"):
        try:
            im, orig_shape = load_image(p, resize=args.resize)
        except Exception:
            print("skip", p); continue
        stem = os.path.splitext(os.path.basename(p))[0]
        token = stem.split('_')[0]
        class_id = class_map.get(token, 0)

        x = tf(im)  # C,H,W tensor
        # build embedding map (C_embed x H x W)
        if embedding_weights is not None:
            if isinstance(embedding_weights, torch.Tensor):
                if class_id < embedding_weights.shape[0]:
                    emb_vec = embedding_weights[class_id].numpy()
                else:
                    emb_vec = np.zeros((EMB_DIM,), dtype=np.float32)
            else:
                emb_vec = np.zeros((EMB_DIM,), dtype=np.float32)
        else:
            emb_vec = np.zeros((EMB_DIM,), dtype=np.float32)

        H = args.resize; W = args.resize
        emb_map = np.tile(emb_vec.reshape(EMB_DIM,1,1), (1,H,W)).astype(np.float32)
        emb_tensor = torch.from_numpy(emb_map)

        # concat channels
        inp = torch.cat([x, emb_tensor], dim=0)  # expect IN_CH == 3+EMB_DIM
        # if mismatch, pad or trim
        if inp.shape[0] < IN_CH:
            pad = torch.zeros((IN_CH - inp.shape[0], H, W), dtype=inp.dtype)
            inp = torch.cat([inp, pad], dim=0)
        elif inp.shape[0] > IN_CH:
            inp = inp[:IN_CH,:,:]

        batch_imgs.append(inp)
        batch_names.append(stem)
        batch_orig.append(orig_shape)

        if len(batch_imgs) >= args.batch_size:
            bt = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                out = model(bt)
                if isinstance(out, dict) and 'out' in out:
                    logits = out['out']
                else:
                    logits = out
                if logits.shape[1]==1:
                    probs = torch.sigmoid(logits[:,0,:,:]).cpu().numpy()
                else:
                    probs_all = F.softmax(logits, dim=1).cpu().numpy()
                    probs = probs_all[:,1:,...].max(axis=1) if probs_all.shape[1]>1 else probs_all[:,0,...]
            for name, prob_map, orig in zip(batch_names, probs, batch_orig):
                prob_resized = cv2.resize((prob_map*255).astype('uint8'), (orig[1], orig[0]), interpolation=cv2.INTER_LINEAR)/255.0
                mask = postprocess_mask(prob_resized, thresh=args.threshold, morph=args.morph, min_area=args.min_area)
                out_mask = os.path.join(args.out_dir, name + ".png")
                cv2.imwrite(out_mask, mask)
                if args.save_prob:
                    np.save(os.path.splitext(out_mask)[0] + ".npy", prob_resized.astype(np.float32))
            batch_imgs=[]; batch_names=[]; batch_orig=[]

    # final batch
    if batch_imgs:
        bt = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            out = model(bt)
            if isinstance(out, dict) and 'out' in out:
                logits = out['out']
            else:
                logits = out
            if logits.shape[1]==1:
                probs = torch.sigmoid(logits[:,0,:,:]).cpu().numpy()
            else:
                probs_all = F.softmax(logits, dim=1).cpu().numpy()
                probs = probs_all[:,1:,...].max(axis=1) if probs_all.shape[1]>1 else probs_all[:,0,...]
        for name, prob_map, orig in zip(batch_names, probs, batch_orig):
            prob_resized = cv2.resize((prob_map*255).astype('uint8'), (orig[1], orig[0]), interpolation=cv2.INTER_LINEAR)/255.0
            mask = postprocess_mask(prob_resized, thresh=args.threshold, morph=args.morph, min_area=args.min_area)
            out_mask = os.path.join(args.out_dir, name + ".png")
            cv2.imwrite(out_mask, mask)
            if args.save_prob:
                np.save(os.path.splitext(out_mask)[0] + ".npy", prob_resized.astype(np.float32))

    print("Saved masks to", args.out_dir)

if __name__ == '__main__':
    main()