#!/bin/bash

set -e  # fail script if any command fails

conda env create -f enviroment.yaml
conda activate prob-specs

# Pull and Install auto_LiRPA
git submodule update --init
cd auto_LiRPA
python setup.py install
# Install CUDA extensions, currently not working
# python auto_LiRPA/cuda_utils.py install
cd ..

