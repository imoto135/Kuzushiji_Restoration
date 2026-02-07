# 🔧 Environment Setup Guide

This guide provides detailed instructions for setting up the development environment for the Kuzushiji Restoration project.

---

## 📋 Prerequisites

- **OS**: Linux (Ubuntu 18.04+) or macOS
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3090 or better)
- **CUDA**: 11.1 or later
- **Python**: 3.7 - 3.10
- **Conda**: Miniconda or Anaconda

---

## 🐍 Environment Installation

### Option 1: NAFNet/MPRNet/SwinIR Environment (Python 3.10)

```bash
cd Kuzushiji_Restoration
conda env create -f environments/env_nafnet2.yml
conda activate nafnet2
```

**Installed packages:**
- PyTorch 1.12.0+cu113
- torchvision 0.13.0+cu113
- BasicSR
- opencv-python
- wandb
- tensorboard

### Option 2: Restormer Environment (Python 3.7)

```bash
cd Kuzushiji_Restoration
conda env create -f environments/environment.yml
conda activate restormer_env
```

**Installed packages:**
- PyTorch 1.9.0+cu111
- torchvision 0.10.0+cu111
- einops
- timm
- fvcore (for FLOPs calculation)

---

## 🔍 Verify Installation

### Check CUDA and GPU

```bash
nvidia-smi
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch version: 1.12.0+cu113
CUDA available: True
CUDA version: 11.3
GPU count: 2
```

### Check BasicSR Installation

```bash
python -c "import basicsr; print(f'BasicSR version: {basicsr.__version__}')"
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Version Mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**
```bash
# Reinstall PyTorch with correct CUDA version
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Issue 2: wandb Login Required

**Symptom:**
```
wandb: ERROR Please login to use wandb
```

**Solution:**
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### Issue 3: Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce batch size in config file
2. Use gradient accumulation
3. Enable mixed precision training (`use_amp: true`)

### Issue 4: Permission Denied

**Symptom:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Make sure you have write permissions
chmod -R 755 experiments/
chmod -R 755 outputs/
```

---

## 📦 Additional Dependencies

### For Evaluation Scripts

```bash
pip install lpips  # For perceptual metrics
pip install pytorch-fid  # For FID score
pip install scikit-image  # For NIQE
```

### For Visualization

```bash
pip install matplotlib
pip install seaborn
pip install pillow
```

---

## 🔄 Environment Management

### List all environments

```bash
conda env list
```

### Activate environment

```bash
conda activate nafnet2  # or restormer_env
```

### Deactivate environment

```bash
conda deactivate
```

### Remove environment

```bash
conda env remove -n nafnet2
```

### Update environment

```bash
conda env update -f environments/env_nafnet2.yml --prune
```

---

## 💾 Disk Space Requirements

| Component | Size | Location |
|-----------|------|----------|
| Dataset (hiragana_fulldataset_5stain) | ~3 GB | `data/` |
| Model weights (per model) | ~500 MB | `models/*/experiments/` |
| Training logs and checkpoints | ~50-100 GB | `models/*/experiments/` |
| Output images | ~1-5 GB | `outputs/` |
| **Total (per model)** | **~55-110 GB** | - |

**Recommendation**: Use external storage or server with sufficient space.

---

## 🖥️ Hardware Recommendations

### Minimum Requirements

- GPU: NVIDIA GTX 1080 Ti (11 GB VRAM)
- RAM: 16 GB
- Storage: 200 GB free space

### Recommended Setup

- GPU: NVIDIA RTX 3090 / A6000 (24-48 GB VRAM)
- RAM: 32 GB or more
- Storage: 500 GB+ SSD

### Training Time Estimates

| Model | GPU | Batch Size | Time (200k iters) |
|-------|-----|------------|-------------------|
| NAFNet | RTX A6000 | 8 | ~3 days |
| SwinIR | RTX A6000 | 8 | ~3 days |
| Restormer | RTX A6000 | 8 | ~3 days |
| MPRNet | RTX A6000 | 64 | ~2 days (50k) |

---

## 🔗 Useful Links

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [BasicSR Documentation](https://basicsr.readthedocs.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)

---

## ✅ Setup Checklist

- [ ] Conda environment created
- [ ] PyTorch with CUDA installed
- [ ] GPU detected by PyTorch
- [ ] BasicSR installed
- [ ] wandb configured
- [ ] Dataset downloaded and placed in `data/`
- [ ] Sufficient disk space available
- [ ] Test run completed successfully

---

**Next Step**: See [TRAINING.md](TRAINING.md) for training instructions.
