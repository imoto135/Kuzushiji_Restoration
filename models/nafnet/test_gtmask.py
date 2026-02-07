#!/usr/bin/env python3
"""
GT Mask互換性テストスクリプト
PairedImageMaskDatasetがGT maskを正しく読み込めるかテスト
"""
import sys
import os
sys.path.insert(0, '/home/imoto/Kuzushiji_Restoration/nafnet')

from basicsr.utils.options import parse
from basicsr.data import create_dataset

def test_gtmask_loading():
    """GT mask設定ファイルでデータセットをロードしてテスト"""
    
    print("="*60)
    print("GT Mask Dataset Loading Test")
    print("="*60)
    
    config_path = '/home/imoto/Kuzushiji_Restoration/nafnet/options/Kuzushiji/gtmask.yml'
    
    # 設定ファイルを読み込み
    opt = parse(config_path, is_train=True)
    
    # Train datasetを作成
    train_opt = opt['datasets']['train']
    print(f"\nTrain Dataset Configuration:")
    print(f"  GT path: {train_opt['dataroot_gt']}")
    print(f"  LQ path: {train_opt['dataroot_lq']}")
    print(f"  Mask path: {train_opt['dataroot_mask']}")
    print(f"  Concat mask: {train_opt.get('concat_mask', False)}")
    
    print(f"\nCreating train dataset...")
    train_dataset = create_dataset(train_opt)
    print(f"✓ Train dataset created: {len(train_dataset)} samples")
    
    # 最初のサンプルを取得
    print(f"\nLoading first sample...")
    sample = train_dataset[0]
    
    lq_shape = sample['lq'].shape
    gt_shape = sample['gt'].shape
    
    print(f"\n✓ Sample loaded successfully:")
    print(f"  LQ shape: {lq_shape}")
    print(f"  GT shape: {gt_shape}")
    
    if 'mask_path' in sample:
        print(f"  Mask path: {sample['mask_path']}")
    
    # LQが4チャンネルかチェック（マスク結合されているか）
    expected_channels = 4 if train_opt.get('concat_mask', False) else 3
    
    if lq_shape[0] == expected_channels:
        print(f"\n✓ LQ channels correct: {lq_shape[0]} (expected: {expected_channels})")
        if expected_channels == 4:
            print(f"  - RGB channels (0-2): shape {lq_shape}")
            print(f"  - Mask channel (3): will be checked")
            
            # マスクチャンネルの値の範囲をチェック
            import torch
            mask_channel = sample['lq'][3]
            print(f"  - Mask channel range: [{mask_channel.min():.4f}, {mask_channel.max():.4f}]")
            print(f"  - Mask channel unique values (first 10): {torch.unique(mask_channel)[:10]}")
    else:
        print(f"\n✗ ERROR: LQ channels mismatch: {lq_shape[0]} (expected: {expected_channels})")
        return False
    
    if gt_shape[0] != 3:
        print(f"\n✗ ERROR: GT channels should be 3, got: {gt_shape[0]}")
        return False
    
    print(f"\n✓ GT channels correct: {gt_shape[0]}")
    
    # バリデーションデータセットもテスト
    print(f"\n{'='*60}")
    print(f"Testing Validation Dataset")
    print(f"{'='*60}")
    
    val_opt = opt['datasets']['val']
    print(f"\nVal Dataset Configuration:")
    print(f"  GT path: {val_opt['dataroot_gt']}")
    print(f"  LQ path: {val_opt['dataroot_lq']}")
    print(f"  Mask path: {val_opt['dataroot_mask']}")
    print(f"  Concat mask: {val_opt.get('concat_mask', False)}")
    
    print(f"\nCreating val dataset...")
    val_dataset = create_dataset(val_opt)
    print(f"✓ Val dataset created: {len(val_dataset)} samples")
    
    print(f"\nLoading first val sample...")
    val_sample = val_dataset[0]
    
    val_lq_shape = val_sample['lq'].shape
    val_gt_shape = val_sample['gt'].shape
    
    print(f"\n✓ Val sample loaded successfully:")
    print(f"  LQ shape: {val_lq_shape}")
    print(f"  GT shape: {val_gt_shape}")
    
    print(f"\n{'='*60}")
    print(f"✅ All Tests Passed!")
    print(f"GT masks are compatible and working correctly.")
    print(f"{'='*60}")
    
    return True

if __name__ == '__main__':
    try:
        success = test_gtmask_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ Test Failed with error:")
        print(f"{str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
