# MPRNet - Multi-Stage Progressive Image Restoration

This directory contains the MPRNet model implementation for Kuzushiji restoration.

## Files

- `MPRNet_arch.py` - MPRNet architecture implementation
- `mprnet_model.py` - MPRNet training model wrapper for BasicSR

## Training Configuration

Located in: `../../configs/nafnet/Kuzushiji/mprnet_mask_charb_percep.yml`

## Model Details

- **Architecture**: Multi-stage progressive restoration with encoder-decoder structure
- **Input**: 4-channel (RGB + predicted mask)
- **Output**: 3-channel RGB
- **Parameters**: 11.9M
- **Training**: 50k iterations with Charbonnier + Perceptual Loss

## Results

| Metric | Value | Iteration |
|--------|-------|-----------|
| PSNR | 37.12 dB | 45k-50k |
| SSIM | - | - |

## Training Command

```bash
cd ../../models/nafnet
python basicsr/train.py -opt options/Kuzushiji/mprnet_mask_charb_percep.yml
```

## Reference

Zamir et al., "Multi-Stage Progressive Image Restoration", CVPR 2021
- Paper: https://arxiv.org/abs/2102.02808
- Original Code: https://github.com/swz30/MPRNet
