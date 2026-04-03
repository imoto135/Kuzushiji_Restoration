FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace/Kuzushiji_Restoration

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1

RUN pip install \
    numpy==1.26.4 \
    ipython>=8.0 \
    addict>=2.4.0 \
    future>=0.18.2 \
    lmdb>=1.3.0 \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.66.0 \
    scipy>=1.13.0 \
    scikit-image>=0.22.0 \
    matplotlib>=3.8.0 \
    lpips>=0.1.4 \
    thop>=0.1.1 \
    wandb>=0.18.0 \
    einops>=0.8.0 \
    timm>=1.0.0 \
    segmentation-models-pytorch>=0.3.3 \
    albumentations>=1.3.0 \
    pandas>=1.5.0 \
    seaborn>=0.13.0

COPY . /workspace/Kuzushiji_Restoration

RUN python3 - <<'PY'
from pathlib import Path
for path in [
    Path('data'),
    Path('outputs'),
    Path('wandb'),
    Path('models/unet++/experiments'),
    Path('models/nafnet/experiments'),
    Path('models/restormer/experiments'),
    Path('models/swinir/experiments'),
    Path('models/joint/experiments'),
    Path('models/classifier/experiments'),
]:
    path.mkdir(parents=True, exist_ok=True)
PY

CMD ["bash"]
