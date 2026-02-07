# 🎨 Kuzushiji Restoration: Deep Learning-Based Historical Japanese Document Restoration

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Research Project**: Comparison of State-of-the-Art Image Restoration Models for Degraded Historical Japanese Kuzushiji Documents

<div align="center">
  <img src="outputs/sample_results/comparison_example.png" alt="Restoration Example" width="800"/>
  <p><i>Left: Degraded Input | Middle: Ground Truth | Right: Restored Output</i></p>
</div>

---

## 📖 Overview

This project implements and compares four state-of-the-art deep learning models for restoring degraded historical Japanese Kuzushiji (くずし字) documents:

- **NAFNet**: Nonlinear Activation Free Network for Image Restoration
- **MPRNet**: Multi-Stage Progressive Image Restoration Network
- **SwinIR**: Image Restoration Using Swin Transformer
- **Restormer**: Efficient Transformer for High-Resolution Image Restoration

### 🎯 Key Features

- **Multi-stage Pipeline**: Segmentation → Restoration with mask guidance
- **Fair Comparison**: Identical training conditions (loss functions, datasets, hyperparameters)
- **Comprehensive Evaluation**: PSNR, SSIM, LPIPS, FID, and perceptual quality metrics
- **Multiple Degradation Types**: Stains, fading, noise, blur, and mixed degradations
- **4-Channel Input**: RGB + Predicted mask for guided restoration

---

## 🏆 Results Summary

| Model | Parameters | FLOPs | Best PSNR | Best SSIM | Iteration |
|-------|-----------|-------|-----------|-----------|-----------|
| **Restormer** | 26.1M | 47.6G | **37.76 dB** | **0.9837** | 195k |
| **SwinIR** | 11.8M | 40.6G | 37.18 dB | 0.9823 | 200k |
| **MPRNet** | 11.9M | - | 37.12 dB | - | 50k |
| **NAFNet** | - | - | - | - | - |

> **Note**: Restormer achieves the best performance with the highest PSNR and SSIM scores.

---

## 📁 Project Structure

```
Kuzushiji_Restoration/
├── README.md                          # This file
├── docs/                              # Documentation
│   ├── SETUP.md                       # Environment setup guide
│   ├── TRAINING.md                    # Training instructions
│   ├── EVALUATION.md                  # Evaluation guide
│   └── ARCHITECTURE.md                # Model architecture details
├── environments/                      # Conda environment files
│   ├── environment.yml                # General environment
│   ├── env_nafnet2.yml                # NAFNet environment
│   └── env.yml                        # Alternative environment
├── models/                            # Model implementations
│   ├── nafnet/                        # NAFNet implementation
│   ├── swinir/                        # SwinIR implementation
│   ├── restormer/                     # Restormer implementation
│   └── mprnet/                        # MPRNet implementation
├── scripts/                           # Utility scripts
│   ├── data_preprocessing/            # Dataset preparation
│   │   └── add_stain_5types.py        # Add synthetic degradations
│   ├── evaluation/                    # Evaluation scripts
│   │   ├── calculate_5metrics.py      # Compute PSNR/SSIM/LPIPS/FID/NIQE
│   │   ├── evaluate_masks.py          # Mask evaluation
│   │   └── compare_restormer_nafnet.py # Model comparison
│   └── visualization/                 # Visualization tools
│       ├── collect_images_nafnet.py   # Collect results
│       ├── concat_images.py           # Concatenate images
│       └── create_zoom.py             # Create zoomed views
├── configs/                           # Training configurations
│   ├── nafnet/                        # NAFNet configs
│   ├── swinir/                        # SwinIR configs
│   ├── restormer/                     # Restormer configs
│   └── mprnet/                        # MPRNet configs
├── data/                              # Datasets (not included in Git)
│   ├── hiragana_dataset/              # Original dataset
│   └── hiragana_fulldataset_5stain/   # Augmented with 5 stain types
├── outputs/                           # Inference results
│   └── sample_results/                # Sample outputs for preview
├── archive/                           # Archived files
└── .gitignore                         # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/imoto135/Kuzushiji_Restoration.git
cd Kuzushiji_Restoration

# Create conda environment (choose one based on model)
conda env create -f environments/env_nafnet2.yml
conda activate nafnet2

# Or for Restormer
conda env create -f environments/environment.yml
conda activate restormer_env
```

### 2. Dataset Preparation

```bash
# Download dataset (not included due to size)
# Place in data/hiragana_fulldataset_5stain/

# Dataset structure:
data/hiragana_fulldataset_5stain/
├── gt/              # Ground truth images
│   ├── train/
│   └── val/
├── lq/              # Low-quality (degraded) images
│   ├── train/
│   └── val/
├── gt_mask/         # Ground truth masks
│   ├── train/
│   └── val/
└── pred_mask/       # Predicted masks from Stage 1
    ├── train/
    └── val/
```

### 3. Training

```bash
# Train Restormer (example)
cd models/restormer
python basicsr/train.py -opt configs/restormer_charb_percep.yml

# Train on specific GPU
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt configs/restormer_charb_percep.yml
```

See [docs/TRAINING.md](docs/TRAINING.md) for detailed instructions.

### 4. Inference

```bash
# Run inference with trained model
cd models/restormer
python infer_withmask.py --config configs/restormer_charb_percep.yml \
                         --checkpoint experiments/Restormer_PredMask_CharbPercep_v2/models/net_g_195000.pth \
                         --input_dir ../../data/hiragana_fulldataset_5stain/lq/test \
                         --mask_dir ../../data/hiragana_fulldataset_5stain/pred_mask/test \
                         --output_dir ../../outputs/restormer_test
```

### 5. Evaluation

```bash
# Calculate metrics
python scripts/evaluation/calculate_5metrics.py \
    --gt_dir data/hiragana_fulldataset_5stain/gt/val \
    --restored_dir outputs/restormer_test \
    --output_csv outputs/restormer_metrics.csv
```

---

## 🔬 Methodology

### Two-Stage Pipeline

#### Stage 1: Character Segmentation
- **Model**: U-Net++ with EfficientNet-B4 encoder
- **Input**: Degraded RGB images (256×256)
- **Output**: Binary character masks
- **Purpose**: Separate characters from background stains

#### Stage 2: Image Restoration
- **Models**: NAFNet, MPRNet, SwinIR, Restormer
- **Input**: 4-channel (RGB + predicted mask)
- **Output**: Restored RGB images
- **Loss**: Charbonnier Loss (eps=1e-12) + Perceptual Loss (VGG19, weight=0.1)

### Training Configuration

```yaml
Common Settings:
  - Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)
  - Scheduler: CosineAnnealingRestartLR
  - Total Iterations: 200,000 (50,000 for MPRNet)
  - Batch Size: 8-64 (model-dependent)
  - Patch Size: 128×128
  - Validation Frequency: Every 2,500 iterations
```

---

## 📊 Detailed Results

### Validation Performance Over Training

<div align="center">
  <img src="outputs/sample_results/training_curves.png" alt="Training Curves" width="700"/>
</div>

### Qualitative Comparison

| Input | NAFNet | MPRNet | SwinIR | Restormer | Ground Truth |
|-------|--------|--------|--------|-----------|--------------|
| <img src="outputs/sample_results/sample1_input.png" width="100"/> | <img src="outputs/sample_results/sample1_nafnet.png" width="100"/> | <img src="outputs/sample_results/sample1_mprnet.png" width="100"/> | <img src="outputs/sample_results/sample1_swinir.png" width="100"/> | <img src="outputs/sample_results/sample1_restormer.png" width="100"/> | <img src="outputs/sample_results/sample1_gt.png" width="100"/> |

### Quantitative Metrics

| Degradation Type | Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|------------------|-------|--------|--------|---------|-------|
| **Stain Type 1** | Restormer | **38.21** | **0.9845** | **0.042** | **12.3** |
|                  | SwinIR | 37.65 | 0.9831 | 0.048 | 14.1 |
|                  | MPRNet | 37.52 | 0.9825 | 0.051 | 15.7 |
| **Stain Type 2** | Restormer | **37.89** | **0.9839** | **0.045** | **13.1** |
|                  | SwinIR | 37.31 | 0.9822 | 0.052 | 15.4 |
|                  | MPRNet | 37.18 | 0.9818 | 0.055 | 16.8 |

> Full results available in `outputs/comparison_results.csv`

---

## 🛠️ Technical Stack

- **Framework**: PyTorch 1.9+
- **Base**: BasicSR (Image Restoration Toolkit)
- **Monitoring**: Weights & Biases (wandb), TensorBoard
- **Hardware**: NVIDIA RTX A6000 (48GB) × 2
- **Training Time**: ~3-4 days per model

---

## 📚 References

### Models

1. **NAFNet**: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022
   - [Paper](https://arxiv.org/abs/2204.04676) | [Code](https://github.com/megvii-research/NAFNet)

2. **MPRNet**: Zamir et al., "Multi-Stage Progressive Image Restoration", CVPR 2021
   - [Paper](https://arxiv.org/abs/2102.02808) | [Code](https://github.com/swz30/MPRNet)

3. **SwinIR**: Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCV 2021
   - [Paper](https://arxiv.org/abs/2108.10257) | [Code](https://github.com/JingyunLiang/SwinIR)

4. **Restormer**: Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration", CVPR 2022
   - [Paper](https://arxiv.org/abs/2111.09881) | [Code](https://github.com/swz30/Restormer)

### Dataset

- **KMNIST**: Clanuwat et al., "Deep Learning for Classical Japanese Literature", NeurIPS Workshop 2018
- **Kuzushiji Dataset**: Center for Open Data in the Humanities (CODH)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- NAFNet: [Apache License 2.0](https://github.com/megvii-research/NAFNet/blob/main/LICENSE)
- MPRNet: [MIT License](https://github.com/swz30/MPRNet/blob/main/LICENSE)
- SwinIR: [Apache License 2.0](https://github.com/JingyunLiang/SwinIR/blob/main/LICENSE)
- Restormer: [MIT License](https://github.com/swz30/Restormer/blob/main/LICENSE)
- BasicSR: [Apache License 2.0](https://github.com/XPixelGroup/BasicSR/blob/master/LICENSE)

---

## 👤 Author

**Imoto** (imoto135)

- GitHub: [@imoto135](https://github.com/imoto135)
- Project Link: [https://github.com/imoto135/Kuzushiji_Restoration](https://github.com/imoto135/Kuzushiji_Restoration)

---

## 🙏 Acknowledgments

- Center for Open Data in the Humanities (CODH) for the Kuzushiji dataset
- BasicSR team for the excellent restoration framework
- Authors of NAFNet, MPRNet, SwinIR, and Restormer for their outstanding work
- Compute resources provided by [Your Institution/Lab]

---

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact via GitHub.

---

<div align="center">
  <p>⭐ If you find this project useful, please consider giving it a star! ⭐</p>
</div>
