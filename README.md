# Neural Network Verification and Repair for Probabilistic Specifications

## Setup

- Install Conda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Configure the pytorch installation by editing the `environment.yaml` file.
  Make sure the CUDA version matches your installation or install pytorch without
  GPU support if your system doesn't have a GPU.
  You can find more information on installing pytorch 
  [here](https://pytorch.org/get-started/previous-versions/#v1121)
- Create a new conda environment using
  ```shell
  NAME=YOUR_PREFERRED_ENVIROMENT_NAME
  conda env create -n $NAME -f environment.yaml
  conda env config vars set PYTHONPATH="$(pwd)" -n $NAME 
  ```
- Install `auto_LiRPA` using
  ```shell
  git submodule update --init  # pull auto_LiRPA if not already present
  conda activate $NAME
  cd auto_LiRPA
  pip install .
  ```
Now you are ready to run the experiments in this repository.
You can also run the last two steps by running `bash -i install.sh` 
from the root directory of this repository.

## Experiments
```bash
python scripts/download_dataset.py  # so that downloading doesn't hamper with timeouts
# ./scripts/train_networks.sh  # Optional, only if you want to recreate the networks
# TODO: pop models
./scripts/run_experiments.sh
```
For reproducing the experiments using the FairSquare algorithm, see `fairsquare/README.md`.
For eProVe, see https://github.com/d-corsi/eProVe/.
