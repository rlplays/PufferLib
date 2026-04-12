#!/bin/bash
set -e

# --- System update ---
apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates gnupg lsb-release \
    python3 python3-pip python3-venv python3-dev \
    build-essential git tmux

# --- CUDA (via NVIDIA's apt repo) ---
# Replace 12-8 and ubuntu2204 with your target versions
CUDA_VERSION="12-8"
DISTRO="ubuntu2204"
ARCH="x86_64"

wget https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y --no-install-recommends \
    cuda-toolkit-${CUDA_VERSION} \
    libcudnn9-cuda-12

# --- Clean up ---
rm -f cuda-keyring_1.1-1_all.deb
apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Python packages ---
pip3 install --upgrade pip uv ninja
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio