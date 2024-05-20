# Probabilistic Verification of Neural Networks
This repository contains the code for the paper "Probabilistic Verification of Neural Networks using Branch and Bound".
The probabilistic verification algorithm is implemented in `probspecs/verifier.py`.
The algorithm for computing bounds on a probability is implemented in `probspecs/bounds/probability_bounds.py`.

## Setup

- Install Conda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Configure the Pytorch installation by editing the `environment.yaml` file.
  Make sure the CUDA version matches your installation or install Pytorch without
  GPU support if your system doesn't have a GPU.
  You can find more information on installing pytorch 
  [here](https://pytorch.org/get-started/previous-versions/#v1121)
- Create a new conda environment using
  ```shell
  export NAME=YOUR_PREFERRED_ENVIROMENT_NAME
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

## Running the Experiments
```bash
python scripts/download_dataset.py  # so that downloading doesn't hamper with hard timeouts
# ./scripts/train_networks.sh  # Optional, only if you want to recreate the networks
./scripts/run_experiments.sh
```
Running the experiments takes up to one day.
The experiment script creates a new directory in `experiments/output`.
The name of the directory is a UTC timestamp (YOUR_TIMESTAMP) for when the experiments were started.
For creating tables and plots like in the paper for your run of the experiments, execute
```bash
python scripts/collect_experiment_results.py experiments/output/YOUR_TIMESTAMP
python scipts/compile_experiment_results.py experiments/output/YOUR_TIMESTAMP
```
This creates several `csv` files and `LaTeX` tables inside the `experiments/output/YOUR_TIMESTAMP` directory.
For more details, have a look at the respective scripts.

For reproducing the experiments using the FairSquare algorithm, see `fairsquare/README.md`.
For eProVe, see https://github.com/d-corsi/eProVe/.

### Compare Different Splitting and Branching Heuristics
For recreating the experiments comparing the `SelectProb` (`prob-mass` heuristic in this repository) 
and the `SelectProbLogBounds` (`prob-log-bounds` heuristic), run
```bash
./scripts/compare_heuristics.sh experiments/config/base.yaml  # this is prob-log-bounds
./scripts/compare_heuristics.sh experiments/config/prob-mass.yaml

python scripts/collect_experiment_results.py experiments/output/TIMESTAMP_BASE
python scripts/collect_experiment_results.py experiments/output/TIMESTAMP_PROB_MASS
python scripts/compile_compare_heuristics.py experiments/output/TIMESTAMP_BASE experiments/output/TIMESTAMP_PROB_MASS
```
