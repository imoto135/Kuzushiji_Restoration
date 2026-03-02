import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def process_images_otsu(input_dir: str, output_dir: str, invert: bool = False, ext: str = ".png"):
    """
    Applies Otsu's binarization to images in the input directory and saves the masks.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    # Get all images with the specified extension recursively
    # Case-insensitive extension search workaround
    image_paths = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() == ext.lower():
            image_paths.append(p)
            
    if not image_paths:
        print(f"No files with extension {ext} found in {input_dir}.")
        return

    print(f"Found {len(image_paths)} images. Starting Otsu's binarization...")
    
    for img_path in tqdm(image_paths, desc="Processing Images"):
        # Load image in grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Failed to read image: {img_path}")
            continue
            
        # Apply Otsu's thresholding. 
        # By default, we use THRESH_BINARY_INV so that dark text on light background becomes white (255) mask on black (0) background.
        # If `invert` is specified, it uses THRESH_BINARY.
        if invert:
            threshold_type = cv2.THRESH_BINARY | cv2.THRESH_OTSU
        else:
            threshold_type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            
        _, mask = cv2.threshold(img, 0, 255, threshold_type)
        
        # Determine the relative path to maintain any sub-directory structure
        rel_path = img_path.relative_to(input_path)
        out_file = output_path / rel_path
        
        # Ensure the output directory for this file exists
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the mask
        cv2.imwrite(str(out_file), mask)

    print(f"Finished generating masks! Saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate text masks from text images using Otsu's binarization.")
    parser.add_argument("-i", "--input_root", type=str, required=True, help="Path to the input root directory (contains train/val/test).")
    parser.add_argument("-o", "--output_root", type=str, required=True, help="Path to the output root directory.")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated list of splits to process.")
    parser.add_argument("--invert", action="store_true", help="Instead of making dark text white, make light text white (uses THRESH_BINARY instead of THRESH_BINARY_INV).")
    parser.add_argument("--ext", type=str, default=".png", help="Image extension to search for (e.g., .png, .jpg). Default is .png")
    
    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    
    for split in splits:
        in_dir = os.path.join(args.input_root, split)
        out_dir = os.path.join(args.output_root, split)
        if not os.path.exists(in_dir):
            print(f"Skipping {split}: Directory not found ({in_dir})")
            continue
        print(f"\nProcessing split: {split}")
        process_images_otsu(in_dir, out_dir, args.invert, args.ext)

if __name__ == "__main__":
    main()
