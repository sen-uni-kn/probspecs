#!/bin/bash

set -e  # fail script if any command fails

# Pull and install auto_LiRPA and other dependencies
git submodule update --init

conda create -y -n "probspecs" python=3.10 pip
conda activate probspecs

# Install PyTorch, optionally with CUDA support
if [ -f "/proc/driver/nvidia/version" ]; then  # CUDA installed?
    if [ -z "$CUDA_VERSION" ]; then
        if [ -x "$(command -v nvcc)" ]; then  # Note: this only works with the nvidia-cuda-toolkit installed
            CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
        else
            CUDA_VERSION="10.2"
        fi
    fi

    # Map to the three supported CUDA version for pytorch 1.12.1
    if (( $(echo "$CUDA_VERSION>=11.6" | bc -l) )); then
        CUDA_VERSION="11.6"
    elif (( $(echo "$CUDA_VERSION>=11.3" | bc -l) )); then
        CUDA_VERSION="11.3"
    else
        CUDA_VERSION="10.2"
    fi

    echo "Installing PyTorch with CUDA version $CUDA_VERSION support."
    conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit="$CUDA_VERSION" -c pytorch -c conda-forge
else
    echo "Installing PyTorch without CUDA support."
    conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
fi

# Install probspecs
pip install -e .
