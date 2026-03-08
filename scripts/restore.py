#!/usr/bin/env python3
"""
Unified Image Restoration Script for Kuzushiji
Supports MPRNet, NAFNet, SwinIR, and Restormer.
Allows toggling mask input via command line arguments.
"""

import sys
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import cv2

# Add models/nafnet to sys.path to access basicsr for MPRNet, NAFNet, SwinIR
repo_root = Path(__file__).parent.parent.absolute()
nafnet_path = repo_root / "models" / "nafnet"
if str(nafnet_path) not in sys.path:
    sys.path.insert(0, str(nafnet_path))

# ---------------------------------------------------------
# Restormer Definition (Inline since it's not in basicsr)
# ---------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = nn.functional.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, data_format='channels_first')
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, data_format='channels_first')
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Restormer(nn.Module):
    def __init__(self, 
                 inp_channels=4,
                 out_channels=3, 
                 dim=48,
                 num_blocks=[4, 6, 6, 8], 
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img[:, :3, :, :]

        return out_dec_level1

# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Unified Image Restoration for Kuzushiji")

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["mprnet", "nafnet", "swinir", "restormer"],
                        help="Model architecture to use.")
    parser.add_argument("--use_mask", action="store_true",
                        help="Whether the model expects a mask (4 channels) or not (3 channels).")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights.")
    parser.add_argument("--input_dir", type=str, default="data/full_padded/lq/test",
                        help="Directory containing damaged images.")
    parser.add_argument("--output_dir", type=str, default="outputs/restored",
                        help="Directory to save restored images.")
    parser.add_argument("--mask_dir", type=str, default="data/full_padded/gt_mask/test",
                        help="Directory containing masks (only used if --use_mask is active).")
                        
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for saving images asynchronously.")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 (half precision) for inference.")
                        
    # wandb settings
    parser.add_argument("--use_wandb", action="store_true", help="Record results to wandb")
    parser.add_argument("--wandb_project", type=str, default="Kuzushiji_Restoration", help="wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None, help="wandb tags")

    return parser.parse_args()


# ---------------------------------------------------------
# Core Logic
# ---------------------------------------------------------
def get_model(args, device):
    """Dynamically load the corresponding model architecture."""
    logging.info(f"Loading {args.model_type.upper()} model... (use_mask={args.use_mask})")

    if args.model_type == "mprnet":
        from basicsr.models.archs.MPRNet_arch import MPRNet
        in_c = 4 if args.use_mask else 3
        model = MPRNet(
            in_c=in_c,
            out_c=3,
            n_feat=40,
            scale_unetfeats=20,
            scale_orsnetfeats=16,
            num_cab=4,
            kernel_size=3,
            reduction=4,
            bias=False,
        )
    elif args.model_type == "nafnet":
        from basicsr.models.archs.NAFNet_arch import NAFNet
        img_channel = 4 if args.use_mask else 3
        model = NAFNet(
            img_channel=img_channel,
            width=32,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2]
        )
    elif args.model_type == "swinir":
        from basicsr.models.archs.SwinIR_arch import SwinIR
        in_chans = 4 if args.use_mask else 3
        model = SwinIR(
            upscale=1,
            in_chans=in_chans,
            out_chans=3,
            img_size=128,
            patch_size=1,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2.0,
            upsampler='',         # '' = denoising mode
            resi_connection='1conv'
        )
    elif args.model_type == "restormer":
        inp_channels = 4 if args.use_mask else 3
        model = Restormer(
            inp_channels=inp_channels,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias'
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Load weights
    try:
        checkpoint = torch.load(args.weights, map_location="cpu")
    except FileNotFoundError:
        logging.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean up keys (handle module., _orig_mod., and total_ops)
    new_state_dict = {}
    for k, v in state_dict.items():
        if "total_ops" in k or "total_params" in k:
            continue
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        elif k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    # Model specific strict loading preference based on past scripts
    strict_load = False if args.model_type in ["restormer", "swinir", "mprnet"] else True
    
    try:
        model.load_state_dict(new_state_dict, strict=strict_load)
    except Exception as e:
        logging.warning(f"Failed to load with strict={strict_load}: {e}")
        logging.warning("Retrying with strict=False")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    model = model.to(device)

    logging.info("Model loaded successfully")
    return model

def resolve_mask_path(img_path, mask_dir):
    """
    Find the corresponding GT mask file by removing the damage type suffix for the mask.
    e.g. U+3042_xxx_Stain.jpg -> mask_dir/U+3042_xxx.png
    """
    if mask_dir is None:
        return None
        
    mask_dir_path = Path(mask_dir)
    damage_types = ['_Transparent_Stain', '_Missing', '_Stain', '_Scratch', '_Ghosting']
    
    mask_stem = img_path.stem
    for dt in damage_types:
        if mask_stem.endswith(dt):
            mask_stem = mask_stem[:-len(dt)]
            break
            
    suffix = img_path.suffix
    for candidate in [
        mask_dir_path / img_path.name,                 # Exact match
        mask_dir_path / f"{mask_stem}{suffix}",        # Removed suffix, same ext
        mask_dir_path / f"{mask_stem}.png",            # Removed suffix, .png
    ]:
        if candidate.exists():
            return candidate
            
    return None

def process_image(model, model_type, img_path, mask_path, use_mask, device):
    """Restore a single image."""
    # Load LQ image (BGR -> RGB)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Normalize image [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Setup Mask if needed
    if use_mask:
        if mask_path is not None and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
            
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0).float()
    
    # HWC -> CHW
    img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))).float()

    # Move to device and batch
    img_tensor = img_tensor.unsqueeze(0).to(device)
    if use_mask:
        mask_tensor = mask_tensor.unsqueeze(0).to(device)
        input_tensor = torch.cat([img_tensor, mask_tensor], dim=1)
    else:
        input_tensor = img_tensor

    # Inference (FP16 autocast)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(input_tensor.device.type == "cuda")):
        outputs = model(input_tensor)

    # Output Parsing (Based on specific model behaviours)
    if isinstance(outputs, (list, tuple)):
        if model_type == "mprnet":
            output = outputs[0]  # MPRNet gives [stage3, stage2, stage1]
        elif model_type == "nafnet":
            output = outputs[-1] # NAFNet gives multiple outputs sometimes, take last
        else:
            output = outputs[-1]
    else:
        output = outputs
        
    output = output.float()

    # Tensor to numpy
    output_np = output.squeeze(0).cpu().numpy()
    output_np = np.transpose(output_np, (1, 2, 0))  # CHW -> HWC
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

    # RGB -> BGR
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    return output_bgr


def main():
    args = parse_args()

    # Setup WandB
    if args.use_wandb:
        import wandb
        run_name = args.wandb_name or f"{args.model_type}_{'Mask' if args.use_mask else 'NoMask'}"
        tags = args.wandb_tags or ['inference', args.model_type]
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            job_type="inference",
            tags=tags,
            config=vars(args),
        )

    # Configure Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Configure Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        cudnn.benchmark = True
        cudnn.deterministic = False

    # Configure Paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir) if args.use_mask else None

    # Find Images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
        
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if len(image_files) == 0:
        logging.error(f"No images found in {input_dir}")
        return

    logging.info(f"Found {len(image_files)} images to process")

    # Load Model
    model = get_model(args, device)

    # Processing Loop
    def save_image(args_tuple):
        path, img = args_tuple
        return cv2.imwrite(str(path), img)

    processed_count = 0
    save_executor = ThreadPoolExecutor(max_workers=args.num_workers)
    save_futures = []

    for img_path in tqdm(image_files, desc=f"Restoring images ({args.model_type})"):
        try:
            mask_path = resolve_mask_path(img_path, mask_dir) if args.use_mask else None

            # Process Image
            restored_img = process_image(model, args.model_type, img_path, mask_path, args.use_mask, device)

            # Save Asynchronously
            output_path = output_dir / f"{img_path.stem}_restored.png"
            save_futures.append((output_path, save_executor.submit(save_image, (output_path, restored_img))))
            processed_count += 1

        except Exception as e:
            logging.error(f"Error processing {img_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Wait for saving to complete
    save_errors = 0
    for output_path, future in save_futures:
        if not future.result():
            logging.error(f"Failed to save {output_path}")
            save_errors += 1
            
    save_executor.shutdown(wait=True)
    actual_processed = processed_count - save_errors

    logging.info(f"Restoration completed. {actual_processed}/{len(image_files)} images saved to {output_dir}")

    # WandB tracking
    if args.use_wandb:
        wandb.log({"inference/processed_images": actual_processed})
        wandb.finish()


if __name__ == "__main__":
    main()
