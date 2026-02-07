#!/usr/bin/env python3
"""
Concatenate GT, LQ, NAFNet Restored, and Restormer Restored images side by side for comparison.
Creates a grid showing: GT | LQ | NAFNet | Restormer

Usage:
    python scripts/concatenate_comparison.py --output_dir outputs/comparison
"""

import os
import re
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def extract_base_name(filename):
    """
    Extract base name from filename.
    GT: U+3042_100241706_00006_1_X0668_Y1844.jpg
    LQ: U+3042_100241706_00006_1_X0668_Y1844_Transparent_Stain.jpg
    Restored: same as LQ
    
    Returns the pattern: U+3042_100241706_00006_1_X0668_Y1844
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Match pattern up to X####_Y####
    match = re.search(r'(.+_X\d+_Y\d+)', name)
    if match:
        return match.group(1)
    return name


def find_matching_files(gt_dir, lq_dir, nafnet_dir, restormer_dir=None):
    """Find matching files across all directories."""
    
    # Build lookup dictionaries
    gt_files = {}
    for f in os.listdir(gt_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            base = extract_base_name(f)
            gt_files[base] = os.path.join(gt_dir, f)
    
    lq_files = {}
    for f in os.listdir(lq_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            base = extract_base_name(f)
            lq_files[base] = os.path.join(lq_dir, f)
    
    nafnet_files = {}
    for f in os.listdir(nafnet_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            base = extract_base_name(f)
            nafnet_files[base] = os.path.join(nafnet_dir, f)
    
    restormer_files = {}
    if restormer_dir and os.path.exists(restormer_dir):
        for f in os.listdir(restormer_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                base = extract_base_name(f)
                restormer_files[base] = os.path.join(restormer_dir, f)
    
    # Find common base names
    common_bases = set(gt_files.keys()) & set(lq_files.keys()) & set(nafnet_files.keys())
    if restormer_files:
        common_bases = common_bases & set(restormer_files.keys())
    
    matches = []
    for base in sorted(common_bases):
        match = {
            'base': base,
            'gt': gt_files[base],
            'lq': lq_files[base],
            'nafnet': nafnet_files[base],
        }
        if restormer_files:
            match['restormer'] = restormer_files[base]
        matches.append(match)
    
    return matches


def concatenate_images(image_paths, labels, add_labels=True, padding=5):
    """
    Concatenate multiple images horizontally with optional labels.
    
    Args:
        image_paths: list of image file paths
        labels: list of labels for each image
        add_labels: whether to add text labels
        padding: padding between images
    
    Returns a PIL Image.
    """
    # Load images
    images = [Image.open(p).convert('RGB') for p in image_paths]
    
    # Get dimensions
    heights = [img.height for img in images]
    max_height = max(heights)
    
    # Resize images to same height if needed
    resized_images = []
    for img in images:
        if img.height != max_height:
            new_width = int(img.width * max_height / img.height)
            img = img.resize((new_width, max_height), Image.LANCZOS)
        resized_images.append(img)
    
    # Calculate total width
    total_width = sum(img.width for img in resized_images) + padding * (len(resized_images) - 1)
    
    # Add space for labels if needed
    label_height = 25 if add_labels else 0
    total_height = max_height + label_height
    
    # Create output image
    output = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Paste images
    x_offset = 0
    x_centers = []
    for img in resized_images:
        output.paste(img, (x_offset, label_height))
        x_centers.append(x_offset + img.width // 2)
        x_offset += img.width + padding
    
    # Add labels
    if add_labels:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(output)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for label, x_pos in zip(labels, x_centers):
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((x_pos - text_width // 2, 3), label, fill=(0, 0, 0), font=font)
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Concatenate GT, LQ, and Restored images for comparison')
    parser.add_argument('--gt_dir', type=str, 
                        default='/home/imoto/Kuzushiji_Restoration/hiragana_fulldataset_5stain/gt/test',
                        help='Ground truth directory')
    parser.add_argument('--lq_dir', type=str,
                        default='/home/imoto/Kuzushiji_Restoration/hiragana_fulldataset_5stain/lq/test',
                        help='Low quality (damaged) directory')
    parser.add_argument('--nafnet_dir', type=str,
                        default='/home/imoto/Kuzushiji_Restoration/outputs/nafnet_predmask_baseline',
                        help='NAFNet restored images directory')
    parser.add_argument('--restormer_dir', type=str,
                        default='/home/imoto/Kuzushiji_Restoration/outputs/restormer_restored_predtest',
                        help='Restormer restored images directory')
    parser.add_argument('--output_dir', type=str,
                        default='/home/imoto/Kuzushiji_Restoration/outputs/comparison_baseline',
                        help='Output directory for concatenated images')
    parser.add_argument('--no_labels', action='store_true',
                        help='Disable labels on images')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--no_restormer', action='store_true',
                        help='Skip restormer images (3 images only)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find matching files
    print(f"Scanning directories...")
    print(f"  GT:       {args.gt_dir}")
    print(f"  LQ:       {args.lq_dir}")
    print(f"  NAFNet:   {args.nafnet_dir}")
    if not args.no_restormer:
        print(f"  Restormer: {args.restormer_dir}")
    
    restormer_dir = None if args.no_restormer else args.restormer_dir
    matches = find_matching_files(args.gt_dir, args.lq_dir, args.nafnet_dir, restormer_dir)
    print(f"Found {len(matches)} matching image sets")
    
    if args.max_images:
        matches = matches[:args.max_images]
        print(f"Processing first {len(matches)} images")
    
    # Define labels
    if args.no_restormer:
        labels = ['GT', 'LQ (Damaged)', 'NAFNet']
    else:
        labels = ['GT', 'LQ (Damaged)', 'NAFNet', 'Restormer']
    
    # Process images
    for match in tqdm(matches, desc="Concatenating images"):
        try:
            if args.no_restormer:
                image_paths = [match['gt'], match['lq'], match['nafnet']]
            else:
                image_paths = [match['gt'], match['lq'], match['nafnet'], match['restormer']]
            
            output_img = concatenate_images(
                image_paths,
                labels,
                add_labels=not args.no_labels
            )
            
            # Save with base name
            output_path = os.path.join(args.output_dir, f"{match['base']}_comparison.jpg")
            output_img.save(output_path, quality=95)
            
        except Exception as e:
            print(f"Error processing {match['base']}: {e}")
    
    print(f"\nDone! Saved {len(matches)} comparison images to: {args.output_dir}")


if __name__ == '__main__':
    main()
