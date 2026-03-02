import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_and_pad_images(input_dir: str, output_dir: str, target_size: int = 128, invert: bool = False, ext: str = ".png"):
    """
    Applies Otsu's binarization to find the background color, calculates its median,
    and then resizes/pads the image to the target_size (default 128x128).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    image_paths = []
    # Case-insensitive extension search workaround
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() == ext.lower():
            image_paths.append(p)
            
    if not image_paths:
        print(f"No files with extension {ext} found in {input_dir}.")
        return

    print(f"Found {len(image_paths)} images. Starting processing...")
    
    for img_path in tqdm(image_paths, desc="Padding Images"):
        # Load image in color
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Warning: Failed to read image: {img_path}")
            continue
            
        # Convert to grayscale for Otsu's binarization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding
        # THRESH_BINARY with OTSU will calculate the optimal threshold.
        ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Determine background mask
        # Normally, text is dark and background is light, so background is > ret
        # If invert is True, text is light and background is dark, so background is < ret
        if invert:
            background_mask = gray < ret
        else:
            background_mask = gray > ret
            
        # Calculate median color of the background
        if np.any(background_mask):
            bg_pixels = img[background_mask]
            pad_color = np.median(bg_pixels, axis=0).astype(int).tolist()
        else:
            # Fallback if mask is somehow empty (e.g., solid color image)
            pad_color = [255, 255, 255]
            
        # Get current image dimensions
        h, w = img.shape[:2]
        
        # Calculate scaling factor to fit into target_size while keeping aspect ratio
        scale = min(target_size / h, target_size / w)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        if scale != 1.0:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        else:
            resized_img = img

        # Calculate padding amounts to center the image
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left
        
        # Pad the image
        padded_img = cv2.copyMakeBorder(
            resized_img, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=pad_color
        )
        
        # Determine the relative path to maintain any sub-directory structure
        rel_path = img_path.relative_to(input_path)
        out_file = output_path / rel_path
        
        # 強制的に拡張子を .png に変更して保存する
        out_file = out_file.with_suffix('.png')
        
        # Ensure the output directory for this file exists
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the padded image
        cv2.imwrite(str(out_file), padded_img)

    print(f"Finished processing! Saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Pad images to a specific size (e.g., 128x128) using the median background color (separated by Otsu).")
    parser.add_argument("-i", "--input_root", type=str, required=True, help="Path to the input root directory (contains train/val/test).")
    parser.add_argument("-o", "--output_root", type=str, required=True, help="Path to the output root directory.")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated list of splits to process.")
    parser.add_argument("--size", type=int, default=128, help="Target image size (default: 128)")
    parser.add_argument("--invert", action="store_true", help="Set this flag if the images have light text on dark background.")
    parser.add_argument("--ext", type=str, default=".jpg", help="Image extension to search for (e.g., .png, .jpg). Default is .jpg")
    
    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    
    for split in splits:
        in_dir = os.path.join(args.input_root, split)
        out_dir = os.path.join(args.output_root, split)
        if not os.path.exists(in_dir):
            print(f"Skipping {split}: Directory not found ({in_dir})")
            continue
        print(f"\nProcessing split: {split}")
        process_and_pad_images(in_dir, out_dir, args.size, args.invert, args.ext)

if __name__ == "__main__":
    main()
