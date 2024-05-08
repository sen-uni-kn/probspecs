#!/bin/bash

set -e  # fail script if any command fails

conda env create -f environment.yaml
conda activate prob-specs
conda env config vars set PYTHONPATH="$(pwd)"
conda activate prob-specs

# Pull and Install auto_LiRPA
git submodule update --init
cd auto_LiRPA
python setup.py install
cd ..

