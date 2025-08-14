#!/bin/bash
set -e

# ===== SETTINGS =====
PYTORCH_VER=2.5.1
TORCHVISION_VER=0.20.1
TORCHAUDIO_VER=2.5.1
PYTHON_VER=3.11

# Install deps
sudo apt-get update
sudo apt-get install -y \
    python${PYTHON_VER} python${PYTHON_VER}-dev python${PYTHON_VER}-distutils \
    python3-pip git cmake build-essential libopenblas-dev libblas-dev \
    libeigen3-dev libjpeg-dev zlib1g-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libopenmpi-dev libomp-dev libsndfile1

# Upgrade pip
python${PYTHON_VER} -m pip install --upgrade pip wheel setuptools ninja

# ===== 1. Build PyTorch =====
cd ~
git clone --branch v${PYTORCH_VER} --depth 1 --recursive https://github.com/pytorch/pytorch
cd pytorch

export USE_NCCL=OFF
export USE_DISTRIBUTED=ON
export BUILD_TEST=0
export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="7.2;8.7;8.9"  # Xavier, Orin

python${PYTHON_VER} -m pip install -r requirements.txt

# Build wheel
python${PYTHON_VER} setup.py bdist_wheel
# Wheel will be in dist/
PYTORCH_WHL=$(ls dist/torch-${PYTORCH_VER}*.whl)

# Install locally
python${PYTHON_VER} -m pip install ${PYTORCH_WHL}

# ===== 2. Build TorchVision =====
cd ~
git clone --branch v${TORCHVISION_VER} --depth 1 https://github.com/pytorch/vision
cd vision
python${PYTHON_VER} -m pip install -r requirements.txt
python${PYTHON_VER} setup.py bdist_wheel
TORCHVISION_WHL=$(ls dist/torchvision-${TORCHVISION_VER}*.whl)
python${PYTHON_VER} -m pip install ${TORCHVISION_WHL}

# ===== 3. Build TorchAudio =====
cd ~
git clone --branch v${TORCHAUDIO_VER} --depth 1 https://github.com/pytorch/audio
cd audio
python${PYTHON_VER} -m pip install -r requirements.txt
python${PYTHON_VER} setup.py bdist_wheel
TORCHAUDIO_WHL=$(ls dist/torchaudio-${TORCHAUDIO_VER}*.whl)
python${PYTHON_VER} -m pip install ${TORCHAUDIO_WHL}

# ===== DONE =====
echo "Build completed!"
echo "Wheels are located in:"
echo "  PyTorch:     $PYTORCH_WHL"
echo "  TorchVision: $TORCHVISION_WHL"
echo "  TorchAudio:  $TORCHAUDIO_WHL"
