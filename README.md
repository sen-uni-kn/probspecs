# Probabilistic Verification of Neural Networks
This repository contains the code for the paper "Probabilistic Verification of Neural Networks using Branch and Bound".
Probabilistic verification means mathematically proving or disproving properties of the output distribution of a neural network under an input distribution.

## Examples

For verifying whether a neural network classifier, for example for credit approval, 
satisfies the demographic parity fairness notion, you can use the `probspecs.Verifier` class.
```python
import torch
from probspecs import Verifier, prob, ExternalFunction, ExternalVariable 

# Build the probabilistic specification
x = ExternalVariable("x")  # represents the network input
male = x[:, 0] >= 1.0  # assuming a one hot encoded variable gender at indices 0 - 2
non_male = x[:, 0] <= 0.0
classifier_func = ExternalFunction("classifier", ("x",))  # argument name needs to match the input variable
credit_approved = classifier_func[:, 0] < classifier_func[:, 1]

p_non_male = prob(credit_approved, condition=non_male)
p_male = prob(credit_approved, condition=male)
# This is the final specification that we'd like to verify (demographic parity)
is_fair = p_non_male / p_male > 0.8

verifier = Verifier(
    worker_devices="cpu",  # or "cuda", may be faster if you have a large neural network 
    timeout=900,  # 15 minutes
)

# Load your actual PyTorch neural network
classifier = torch.load(...)
# Load an input distribution. For example, a `probspecs.distributions.BayesianNet`
input_distribution = torch.load(...)
# A description of the input space. Examples in `experiments/input_spaces.py`.
input_space = torch.load(...)

verification_status, probability_bounds = verifier.verify(
    is_fair,
    {"classifier": classifier},  # match specification to actual neural networks
    {"x": input_space},
    {"x": input_distribution},
)
```
The verification_status is either `probspecs.VerifyStatus.SATISFIED` if your
neural network was proven to satisfy the specification or
`probspecs.VerifyStatus.VIOLATED` if the verifier proved that your network
*does not* satisfy the specification.
If the verification times out, the verifier throws a `probspecs.VerifierTimeout` exception.

Another application of probabilistic verification is computing a lower and an upper bound
on a probability.
These bounds are guaranteed to encode the actual value of the probability.
```python
import torch
from probspecs import prob, ExternalFunction, ExternalVariable, TensorInputSpace
from probspecs.bounds import ProbabilityBounds
from probspecs.distributions import Uniform

# Build the probability that you want to compute bounds on.
x = ExternalVariable("x")  # the input
net_func = ExternalFunction("network", ("x",))
output_constraint = net_func[:, 0] >= net_func[:, 1]  # for example
input_constraint = (x[:, 0] >= 0.0) & (x[:, 0] <= 1.0)  # for example
target_probability = prob(~output_constraint, condition=input_constraint)

# Example input space and input distribution
# The input space is 5d and all inputs lie between -1 and 1.
input_space = TensorInputSpace(-torch.ones(5), torch.ones(5))
# Example for a simple input distribution
input_distribution = Uniform(input_space.input_bounds)
network = torch.load(...)

compute_bounds = ProbabilityBounds(
  device="cpu",  # or cuda, or cuda:0, etc
  batch_size=512  # depending on your network you may want to try larger or smaller values
)

# bounds_gen is an infinite generator that generates tighter and tigher bounds.
bounds_gen = compute_bounds.bound(
    target_probability,
    {"network": network},
    {"x": input_space},
    {"x": input_distribution},
)
best_bounds = None
lower, upper = -float("inf"), float("inf")
while True:  # you probably want to implement a timeout here.
    best_bounds = next(bounds_gen)
    lower, upper = best_bounds
    print(f"{lower:.6f} <= P(net_0 >= net_1 | 0 <= x_0 <= 1) <= {upper:.6f}")
```
You can find further examples in the `experiments` directory.
In particular,
 - `experiments/acasxu/safety.py`
 - `experiments/acasxu/robustness.py`
 - `experiments/mini_acs_income/verify.py`

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

## Paper 
This repository allows you to reproduce the experiments from the paper "Probabilistic Verification of Neural Networks using Branch and Bound".

### Repository Layout
 - the probabilistic verification algorithm (`PV`) is implemented in `probspecs/verifier.py`.
 - the algorithm for computing bounds on a probability (`ProbabilityBounds`) is implemented in `probspecs/bounds/probability_bounds.py`.
 - all experiments are located in the `experiments/` folder. Scripts for running the experiments and analysing their output are located in `scipts`.
 - the `MiniACSIncome` datasets are defined in `experiments/mini_acs_income.py`. Neural networks and input distributions are contained in `resources/mini_acs_income`.

### Running the Experiments
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
