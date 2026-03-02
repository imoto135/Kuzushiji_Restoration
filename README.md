# 🎨 Kuzushiji Restoration: Deep Learning-Based Historical Japanese Document Restoration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **研究プロジェクト**: くずし字（崩し字）を含む歴史的日本語文書の劣化修復に向けた、最先端深層学習モデルの比較研究

---

## 📖 概要 / Overview

本プロジェクトでは、シミ・かすれ・スクラッチ等の劣化を受けた歴史的日本語文書（くずし字）の画像復元を目的として、4種の最先端深層学習モデルを実装・比較します。

**解決すべき課題**：
- 歴史的文書は経年劣化（シミ・退色・スクラッチ・透過シミなど5種類）により判読困難になる
- くずし字は現代人には読みにくく、高精度な復元が文化財のデジタル保存・研究に貢献する
- 単純な画像復元モデルでは文字領域と背景劣化を区別できない

**本研究のアプローチ**：
文字セグメンテーション（Stage 1）と画像復元（Stage 2）の**二段階パイプライン**を構築し、文字マスクを補助情報として活用することで、より精密な復元を実現します。

---

## 🏆 Results Summary

| Model | Parameters | Best PSNR | Best SSIM | Best LPIPS | Iterations |
|-------|-----------|-----------|-----------|------------|------------|
| **Restormer** | 26.1M | **37.76 dB** | **0.9837** | **0.0214** | 195k |
| **SwinIR** | 11.8M | 37.18 dB | 0.9823 | 0.0231 | 200k |
| **MPRNet** | 11.9M | 37.12 dB | 0.9819 | 0.0248 | 50k |
| **NAFNet** | 29.1M | *(training in progress)* | — | — | — |

> Restormer が最も高いPSNR・SSIMを達成。TransformerベースのGlobal Attentionが文字の細部復元に有効と考えられます。

---

## 🔬 Methodology

### Two-Stage Pipeline

```
Degraded Image
      │
      ▼
┌─────────────────────────┐
│  Stage 1: Segmentation  │  ← UNet++ (EfficientNet-B4 encoder)
│  Input : RGB (256×256)  │
│  Output: Binary Mask    │
└──────────┬──────────────┘
           │  Predicted Mask
           ▼
┌──────────────────────────────┐
│  Stage 2: Restoration        │  ← NAFNet / MPRNet / SwinIR / Restormer
│  Input : 4-ch (RGB + Mask)   │
│  Output: Restored RGB Image  │
└──────────────────────────────┘
```

#### Stage 1: Character Segmentation
- **モデル**: U-Net++ (EfficientNet-B4 encoder, ImageNet pretrained)
- **入力**: 劣化RGB画像（256×256）
- **出力**: 文字領域の二値マスク
- **目的**: 文字ピクセルと劣化背景ピクセルを分離し、後段の復元モデルに手がかりを与える

#### Stage 2: Image Restoration
- **モデル**: NAFNet / MPRNet / SwinIR / Restormer（4モデルを同条件で比較）
- **入力**: 4チャンネル（RGB 3ch + 予測マスク 1ch）
- **出力**: 復元RGB画像
- **損失関数**: Charbonnier Loss + Perceptual Loss（VGG19, `conv5_4` layer, weight=0.1）

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=1e-3, betas=[0.9, 0.9]) |
| Scheduler | CosineAnnealingRestartLR |
| Batch Size | 64 (NAFNet) / 8 (Restormer) |
| Patch Size | 128×128 |
| Mixed Precision | AMP (fp16) |
| Validation | Every 5,000 iterations |
| Early Stopping | LPIPS-based (patience=10) |

### Degradation Types

学習・評価に使用した5種類の合成劣化：

| Type | 説明 |
|------|------|
| Stain | 局所的なシミ（茶色・黒） |
| Transparent Stain | 半透明の染み |
| Scratch | 線状のスクラッチ |
| Ghosting | 裏写り・ゴースト |
| Missing | 部分的な文字欠損 |

---

## 📁 Project Structure

```
Kuzushiji_Restoration/
├── configs/                           # 学習設定ファイル (YAML)
│   ├── nafnet/Kuzushiji/              # NAFNet 各種実験設定
│   ├── swinir/                        # SwinIR 各種実験設定
│   ├── restormer/                     # Restormer 各種実験設定
│   └── mprnet/                        # MPRNet 各種実験設定
├── models/                            # モデル実装
│   ├── nafnet/                        # NAFNet (BasicSR ベース)
│   ├── swinir/                        # SwinIR (BasicSR ベース)
│   ├── restormer/                     # Restormer (BasicSR ベース)
│   └── mprnet/                        # MPRNet (BasicSR ベース)
├── scripts/
│   ├── data_preprocessing/            # データ前処理
│   │   ├── 01_pad_images.py           # 画像のパディング (128×128)
│   │   ├── 02_generate_otsu_masks.py  # 大津の二値化によるGTマスク生成
│   │   └── add_stain_5types.py        # 5種類の合成劣化付与
│   ├── evaluation/                    # 評価スクリプト
│   │   ├── calculate_5metrics.py      # PSNR / SSIM / LPIPS / FID / NIQE 計算
│   │   └── create_comparison_image.py # 複数モデルの比較画像生成
│   └── restore_with_*.py              # 各モデルの推論スクリプト
├── environments/                      # Conda 環境ファイル
└── data/                              # データセット (Git 管理外)
    └── full_padded/
        ├── gt/   lq/   gt_mask/   pred_mask/
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/imoto135/Kuzushiji_Restoration.git
cd Kuzushiji_Restoration

# NAFNet / SwinIR / Restormer 用
conda env create -f environments/env_nafnet2.yml
conda activate nafnet2
```

### 2. Dataset Preparation

```bash
# データセットは data/full_padded/ 以下に配置
# gt/ : Ground Truth画像
# lq/ : 劣化画像 (5種のaugmentation適用済み)
# gt_mask/   : 大津の二値化によるGTマスク
# pred_mask/ : Stage 1 (UNet++) による予測マスク

python scripts/data_preprocessing/01_pad_images.py
python scripts/data_preprocessing/02_generate_otsu_masks.py
```

### 3. Training

```bash
# 例: Restormer (Charbonnier + Perceptual Loss, Mask入力あり)
cd models/restormer
python basicsr/train.py -opt ../../configs/restormer/Kuzushiji/gtmask_charb_percep.yml

# 例: NAFNet (フルデータセット, Mask入力あり)
cd models/nafnet
python launch_train.py --opt ../../configs/nafnet/Kuzushiji/full_mask_charb_percep.yml
```

### 4. Inference

```bash
# Restormer による推論
python scripts/restore_with_restormer.py \
    --input_dir data/full_padded/lq/test \
    --mask_dir  data/full_padded/pred_mask/test \
    --output_dir outputs/restormer_gtmask

# NAFNet による推論
python scripts/restore_with_nafnet.py \
    --input_dir data/full_padded/lq/test \
    --output_dir outputs/nafnet_test
```

### 5. Evaluation

```bash
python scripts/evaluation/calculate_5metrics.py \
    --gt_dir data/full_padded/gt/test \
    --restored_dir outputs/restormer_gtmask \
    --output_csv outputs/restormer_metrics.csv
```

---

## 🛠️ Technical Stack

| Category | Tool / Library |
|----------|---------------|
| Deep Learning | PyTorch 2.0+, CUDA |
| Framework | BasicSR (Image Restoration Toolkit) |
| Segmentation | segmentation_models_pytorch (UNet++) |
| Metrics | PSNR, SSIM, LPIPS (lpips), FID (pytorch-fid), NIQE |
| Monitoring | Weights & Biases (wandb), TensorBoard |
| Hardware | NVIDIA RTX A6000 (48GB) |
| Environment | Conda, Python 3.9+ |

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

- **Kuzushiji Dataset**: Center for Open Data in the Humanities (CODH)
- Clanuwat et al., "Deep Learning for Classical Japanese Literature", NeurIPS Workshop 2018

---

## 📄 License

This project is licensed under the MIT License.

### Third-Party Licenses

- BasicSR: [Apache License 2.0](https://github.com/XPixelGroup/BasicSR/blob/master/LICENSE)
- NAFNet: [Apache License 2.0](https://github.com/megvii-research/NAFNet/blob/main/LICENSE)
- MPRNet / Restormer: [MIT License](https://github.com/swz30/MPRNet/blob/main/LICENSE)
- SwinIR: [Apache License 2.0](https://github.com/JingyunLiang/SwinIR/blob/main/LICENSE)

---

## 👤 Author

**Imoto** — [@imoto135](https://github.com/imoto135)

Project: [https://github.com/imoto135/Kuzushiji_Restoration](https://github.com/imoto135/Kuzushiji_Restoration)

---

## 🙏 Acknowledgments

- Center for Open Data in the Humanities (CODH) for the Kuzushiji dataset
- BasicSR team for the excellent restoration framework
- Authors of NAFNet, MPRNet, SwinIR, and Restormer for their outstanding work
