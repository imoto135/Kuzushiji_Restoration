#!/usr/bin/env python3
"""
Quick test script to verify both mask and nomask configurations
"""
import sys
import os

# Add basicsr to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'basicsr'))

from basicsr.utils.options import parse
from basicsr.data import create_dataset

def test_config(config_path):
    """Test a configuration file"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_path}")
    print(f"{'='*60}")
    
    try:
        # Parse config
        opt = parse(config_path, is_train=True)
        print(f"✓ Configuration parsed successfully")
        print(f"  - Name: {opt['name']}")
        print(f"  - Model: {opt['model_type']}")
        
        # Check network config
        net_g = opt.get('network_g', {})
        img_channel = net_g.get('img_channel', 3)
        out_channel = net_g.get('out_channel', 3)
        print(f"  - Network input channels: {img_channel}")
        print(f"  - Network output channels: {out_channel}")
        
        # Test train dataset
        train_opt = opt['datasets']['train']
        print(f"\n  Train Dataset:")
        print(f"    - Type: {train_opt['type']}")
        print(f"    - GT path: {train_opt['dataroot_gt']}")
        print(f"    - LQ path: {train_opt['dataroot_lq']}")
        
        has_mask = 'dataroot_mask' in train_opt
        concat_mask = train_opt.get('concat_mask', False)
        print(f"    - Has mask path: {has_mask}")
        if has_mask:
            print(f"    - Mask path: {train_opt['dataroot_mask']}")
        print(f"    - Concat mask: {concat_mask}")
        
        # Create train dataset (just first sample)
        print(f"\n  Creating train dataset...")
        train_set = create_dataset(train_opt)
        print(f"  ✓ Dataset created: {len(train_set)} samples")
        
        # Get first sample
        print(f"\n  Loading first sample...")
        sample = train_set[0]
        lq_shape = sample['lq'].shape
        gt_shape = sample['gt'].shape
        
        print(f"  ✓ Sample loaded:")
        print(f"    - LQ shape: {lq_shape}")
        print(f"    - GT shape: {gt_shape}")
        
        if 'mask' in sample:
            mask_shape = sample['mask'].shape
            print(f"    - Mask shape: {mask_shape}")
        
        # Verify channel count matches network
        if lq_shape[0] != img_channel:
            print(f"  ✗ ERROR: LQ channels ({lq_shape[0]}) != network input channels ({img_channel})")
            return False
        
        if gt_shape[0] != out_channel:
            print(f"  ✗ ERROR: GT channels ({gt_shape[0]}) != network output channels ({out_channel})")
            return False
        
        # Test val dataset
        val_opt = opt['datasets']['val']
        print(f"\n  Val Dataset:")
        print(f"    - Type: {val_opt['type']}")
        
        has_val_mask = 'dataroot_mask' in val_opt
        concat_val_mask = val_opt.get('concat_mask', False)
        print(f"    - Has mask path: {has_val_mask}")
        if has_val_mask:
            print(f"    - Mask path: {val_opt['dataroot_mask']}")
        print(f"    - Concat mask: {concat_val_mask}")
        
        print(f"\n  Creating val dataset...")
        val_set = create_dataset(val_opt)
        print(f"  ✓ Dataset created: {len(val_set)} samples")
        
        # Get first val sample
        print(f"\n  Loading first val sample...")
        val_sample = val_set[0]
        val_lq_shape = val_sample['lq'].shape
        val_gt_shape = val_sample['gt'].shape
        
        print(f"  ✓ Sample loaded:")
        print(f"    - LQ shape: {val_lq_shape}")
        print(f"    - GT shape: {val_gt_shape}")
        
        if 'mask' in val_sample:
            val_mask_shape = val_sample['mask'].shape
            print(f"    - Mask shape: {val_mask_shape}")
        
        print(f"\n{'='*60}")
        print(f"✓ Configuration test PASSED: {config_path}")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Configuration test FAILED: {config_path}")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return False

if __name__ == '__main__':
    os.chdir('/home/imoto/Kuzushiji_Restoration/nafnet')
    
    configs = [
        'options/Kuzushiji/nomask.yml',
        'options/Kuzushiji/mask.yml',
    ]
    
    print("\n" + "="*60)
    print("NAFNet Configuration Verification Test")
    print("="*60)
    
    results = {}
    for config in configs:
        results[config] = test_config(config)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for config, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {config}")
    
    print("\n")
    all_passed = all(results.values())
    if all_passed:
        print("🎉 All configurations verified successfully!")
        sys.exit(0)
    else:
        print("❌ Some configurations failed - please review errors above")
        sys.exit(1)
