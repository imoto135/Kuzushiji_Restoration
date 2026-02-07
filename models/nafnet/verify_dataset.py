import sys
import os
import yaml
import torch
import cv2

# Run from inside 'nafnet' directory
# So current directory is .../nafnet
sys.path.append(os.getcwd())

from basicsr.data.paired_image_mask_dataset import PairedImageMaskDataset

def verify(config_path):
    print(f"\n--- Verifying {config_path} ---")
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    
    # Train Dataset
    if 'train' in opt['datasets']:
        print("Checking Train Dataset...")
        train_opt = opt['datasets']['train']
        train_opt['phase'] = 'train'
        train_opt['scale'] = opt['scale']
        
        try:
            dataset = PairedImageMaskDataset(train_opt)
            print(f"Train Dataset length: {len(dataset)}")
            
            if len(dataset) > 0:
                # Get item 0
                item = dataset[0]
                print(f"Sample 0 LQ Shape: {item['lq'].shape}")
                print(f"Sample 0 GT Shape: {item['gt'].shape}")
                print(f"Sample 0 LQ Path: {os.path.basename(item['lq_path'])}")
                print(f"Sample 0 GT Path: {os.path.basename(item['gt_path'])}")
                
                if 'mask_path' in item:
                    print(f"Sample 0 Mask Path: {os.path.basename(item['mask_path'])}")
                else:
                    print("Sample 0 Mask Path: None")
                    
                # Verify channel count matching config
                # Check expected channels
                expected_input_ch = 4 if 'mask.yml' in config_path else 3
                if item['lq'].shape[0] != expected_input_ch:
                    print(f"[FAIL] Expected {expected_input_ch} channels, got {item['lq'].shape[0]}")
                else:
                    print(f"[PASS] Channel count correct ({item['lq'].shape[0]})")
                    
        except Exception as e:
            print(f"[ERROR] Failed to load train dataset: {e}")
            import traceback
            traceback.print_exc()

    # Val Dataset
    if 'val' in opt['datasets']:
        print("\nChecking Val Dataset...")
        val_opt = opt['datasets']['val']
        val_opt['phase'] = 'val'
        val_opt['scale'] = opt['scale']
        
        try:
            dataset = PairedImageMaskDataset(val_opt)
            print(f"Val Dataset length: {len(dataset)}")
            if len(dataset) > 0:
                item = dataset[0]
                print(f"Sample 0 LQ Shape: {item['lq'].shape}")
                
        except Exception as e:
            print(f"[ERROR] Failed to load val dataset: {e}")

if __name__ == '__main__':
    verify('options/Kuzushiji/mask.yml')
    verify('options/Kuzushiji/nomask.yml')
