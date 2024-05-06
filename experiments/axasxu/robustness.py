#  Copyright (c) 2024 David Boetius
#  Licensed under the MIT License
import argparse
from time import time
from pathlib import Path

import torch
from tqdm import tqdm

from probspecs import (
    prob,
    ExternalFunction,
    ExternalVariable,
    TensorInputSpace,
    and_expr,
)
from probspecs.bounds import ProbabilityBounds
from probspecs.distributions import Uniform, PointDistribution, MultivariateIndependent
from probspecs.utils.yaml import yaml
from experiments.utils import get_acasxu_network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compute Output Distributions of an ACAS Xu Network under adversarial inputs."
    )
    parser.add_argument(
        "-n",
        "--network",
        choices=tuple(f"{i}_{j}" for i in range(1, 6) for j in range(1, 10)),
        default="1_1",
        help="The ACAS Xu network number of the network to investigate.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=int,
        default=0,
        help="The input for which to investigate local robustness. "
        "If the value of --input is i, this script selects the ith input among"
        " a random sample of inputs (for a fixed seed) "
        "that is classified by the network as --label.",
    )
    parser.add_argument(
        "-l",
        "--label",
        choices=tuple(range(5)),
        type=int,
        required=True,
        help="What label to search for for the input point.",
    )
    parser.add_argument(
        "-t",
        "--target-label",
        choices=tuple(range(5)),
        type=int,
        required=True,
        help="The target label whose probability to bound.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="Size of the input region as a percentage of the input range (per input dimension).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="A timeout for computing bounds on the frequency of property violations "
        "in seconds.",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=1e-3,
        help="A threshold for the precision of the computed bounds. This script stops once this threshold is reached.",
    )
    parser.add_argument(
        "--probability-bounds-config",
        default="{}",
        help="A configuration for computing bounds. Can be a path to a YAML file "
        "or a yaml string. Have a look at the ProbabilityBounds class for details "
        "on which configurations are available.",
    )
    args = parser.parse_args()

    net_i1, net_i2 = args.network.split("_")
    network, (input_lbs, input_ubs) = get_acasxu_network(net_i1, net_i2)
    input_range = input_ubs - input_lbs

    rng = torch.Generator()
    rng.manual_seed(712589555260823)
    reference_input = None
    find_ith = args.input
    print(f"Generate reference input: {find_ith}th input with label {args.label}.")
    while reference_input is None:
        batch = input_lbs + input_range * torch.rand(
            (1024,) + input_lbs.shape, generator=rng
        )
        # Classifier chooses the class with the *minimal* score.
        classes = torch.argmin(network(batch), dim=-1)
        right_label = classes == args.label
        if torch.sum(right_label) == 0:
            selected = []
        else:
            selected = batch[right_label]
        if len(selected) < find_ith:
            find_ith -= len(selected)
        else:
            reference_input = selected[find_ith]
        print(
            f"New Batch: {100*len(selected)/len(batch):3.0f}% label {args.label}. Left to skip: {find_ith}."
        )

    eps = ExternalVariable("eps")
    # x[:, 0]: rho (distance between ownship and intruder)
    # x[:, 1]: theta (angle between ownship heading angle and intruder)
    # x[:, 2]: psi (angle between ownship heading angle and intruder heading angle)
    # x[:, 3]: v_own (ownship speed)
    # x[:, 4]: v_int (intruder speed)
    net_func = ExternalFunction("network", ("x",))

    # Adversarial perturbation: only first two inputs (Converse et al. 2020)
    eps = input_range * args.eps
    rho_lbs = max(reference_input[0] - eps[0], input_lbs[0])
    rho_ubs = max(reference_input[0] + eps[0], input_ubs[0])
    theta_lbs = max(reference_input[1] - eps[1], input_lbs[1])
    theta_ubs = max(reference_input[1] + eps[1], input_ubs[1])

    x_lbs = torch.tensor([rho_lbs, theta_lbs, *reference_input[2:]])
    x_ubs = torch.tensor([rho_ubs, theta_ubs, *reference_input[2:]])
    input_space = TensorInputSpace(x_lbs, x_ubs)

    # Classifier chooses the class with the *minimal* score.
    target = args.target_label
    output_constraint = and_expr(
        *[
            net_func[:, target] < net_func[:, other]
            for other in range(5)
            if other != target
        ]
    )

    p_target = prob(output_constraint)

    # count violations: uniform distribution
    rho_theta_distribution = Uniform((x_lbs[:2], x_ubs[:2]))
    remaining_inputs_distribution = PointDistribution(reference_input[2:])
    input_distribution = MultivariateIndependent(
        rho_theta_distribution, remaining_inputs_distribution, event_shape=(5,)
    )

    timeout = args.timeout
    if timeout is None:
        timeout = float("inf")

    if "{" in args.probability_bounds_config or "\n" in args.probability_bounds_config:
        prob_bounds_config = args.probability_bounds_config
    else:
        prob_bounds_config = Path(args.probability_bounds_config)
    prob_bounds_config = yaml.load(prob_bounds_config)
    prob_bounds_config = {"batch_size": 512} | prob_bounds_config
    compute_bounds = ProbabilityBounds(device="cpu", **prob_bounds_config)

    start_time = time()
    bounds_gen = compute_bounds.bound(
        p_target,
        {"network": network},
        {"x": input_space},
        {"x": input_distribution},
    )
    best_bounds = None
    while (time() - start_time) < timeout:
        best_bounds = next(bounds_gen)
        lower, upper = best_bounds
        print(f"{lower:.6f} <= P(net(x) = {target}) <= {upper:.6f}")
        if upper - lower <= args.precision:
            break
    runtime = time() - start_time
    print(f"Finished. Runtime {runtime:.2f}s")
