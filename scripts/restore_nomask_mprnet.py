import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import sys

# Add model path
mprnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/mprnet')
sys.path.insert(0, mprnet_path)

from MPRNet_arch import MPRNet


def img2tensor(img):
    """HWC uint8 RGB numpy -> CHW float [0,1] tensor"""
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


def tensor2img(tensor):
    """CHW float [0,1] tensor -> HWC uint8 RGB numpy"""
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def load_model(model_path, device='cuda'):
    """Load MPRNet model from checkpoint."""
    model = MPRNet(n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=4)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both direct state_dict and nested state_dict
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def restore_image(model, img_path, device='cuda', output_path=None):
    """Restore a single image using MPRNet."""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read image: {img_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Prepare tensor
    img_tensor = img2tensor(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        
    if isinstance(output, list):
        output = output[0]  # MPRNet outputs [stage3, stage2, stage1]
    
    # Convert output to image
    output_img = tensor2img(output)
    
    # Convert RGB to BGR for saving
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(str(output_path), output_img)
        print(f"Saved: {output_path}")
    
    return output_img


def main():
    parser = argparse.ArgumentParser(description='Restore images using MPRNet')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('-m', '--model', type=str, 
                        default='models/mprnet/experiments/MPRNet_Kuzushiji_NoMask_CharbPercep/models/net_g_220000.pth',
                        help='Model checkpoint path')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    model = load_model(args.model, device)
    print("Model loaded successfully")
    
    # Handle input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        output_file = os.path.join(args.output, input_path.stem + '_restored.png')
        restore_image(model, input_path, device, output_file)
    
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.glob('*') 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images")
        
        for idx, img_file in enumerate(image_files, 1):
            output_file = os.path.join(args.output, img_file.stem + '_restored.png')
            if os.path.exists(output_file):
                print(f"[{idx}/{len(image_files)}] Skipped (already exists): {output_file}")
                continue
            restore_image(model, img_file, device, output_file)
            print(f"[{idx}/{len(image_files)}] Processed")
    
    else:
        print(f"Error: Input path not found: {args.input}")


if __name__ == '__main__':
    main()