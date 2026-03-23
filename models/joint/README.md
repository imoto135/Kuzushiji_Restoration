# models/joint — End-to-End Joint UNet++/NAFNet Restoration

This directory contains the **joint end-to-end training** framework that integrates Stage 1 (UNet++) and Stage 2 (NAFNet) into a single differentiable pipeline.

## Files

| File | Description |
|---|---|
| `joint_model.py`  | `JointRestorationNet`: UNet++ → soft mask → NAFNet, all in one `nn.Module` |
| `joint_dataset.py`| `KuzushijiJointDataset`: returns `(lq, gt, gt_mask)` triplets |
| `train.py`        | Standalone training script (no BasicSR required) |
| `inference.py`    | Batch inference script |
| `options/Kuzushiji/joint_charb_percep.yml` | Training config |

## Architecture

```
LQ (B,3,H,W)
   │
   ├─► ImageNet-normalize ─► UNet++ (SE-ResNeXt-50) ─► logits ─► sigmoid ─► soft mask (B,1,H,W)
   │                                                                               │
   └──────────── cat ──────────────────────────────────────────────────────────────┘
                  │
                 (B,4,H,W)
                  │
                NAFNet (width=32, img_channel=4)
                  │
             restored (B,3,H,W)
```

## Loss Function

```
L_total = L_restore + λ_seg * L_seg

L_restore = CharbonnierLoss + λ_percep * PerceptualLoss(VGG19)
L_seg     = 0.5 * DiceLoss + 0.5 * BCELoss
```

Default: `λ_seg = 0.1`, `λ_percep = 0.1`

## Training

```bash
# Activate the existing NAFNet environment
conda activate nafnet2

cd /home/imoto/Kuzushiji_Restoration/models/joint

# Full training (100 epochs, with WandB)
python train.py --config options/Kuzushiji/joint_charb_percep.yml

# Quick smoke test (5 epochs, no WandB)
python train.py --config options/Kuzushiji/joint_charb_percep.yml \
                --epochs 5 --no_wandb
```

## Inference

```bash
python inference.py \
    --model_path experiments/joint_unetpp_nafnet_charb_percep/best_model.pth \
    --input_dir  ../../data/hiragana_fulldataset_5stain/lq/test \
    --output_dir ../../outputs/joint_test \
    --save_mask
```

## Evaluation

Use the existing evaluation script:

```bash
python ../../scripts/evaluation/calculate_5metrics.py \
    --gt_dir      ../../data/hiragana_fulldataset_5stain/gt/test \
    --restored_dir ../../outputs/joint_test \
    --output_csv  ../../outputs/joint_metrics.csv
```
