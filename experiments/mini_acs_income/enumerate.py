# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import argparse
import itertools
from pathlib import Path
from time import time

import torch
from miniacsincome import MiniACSIncome, get_network, get_population_model
from torchstats import TabularInputSpace

from experiments.utils import log_machine_and_code_details
from probspecs import VerifyStatus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Verify Group Fairness by Testing all Discrete Values"
    )
    parser.add_argument(
        "-v",
        "--num-variables",
        choices=tuple(range(1, 9)),
        type=int,
        required=True,
        help="The number of input variables. "
        "Determines the complexity of the verification problem.",
    )
    parser.add_argument("-n", "--network", default="network")
    parser.add_argument("--fairness-eps", default=0.2)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="A timeout for computing bounds on the frequency of property violations "
        "in seconds.",
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether to print progress messages."
    )
    args = parser.parse_args()

    print("Running Experiment: MiniACSIncome - Enumerate the Discrete Values")
    print("=" * 100)
    print("Command Line Arguments:")
    print(args)
    log_machine_and_code_details()

    dataset = MiniACSIncome(
        root=".datasets", num_variables=args.num_variables, download=True
    )

    input_space: TabularInputSpace
    input_distribution, input_space, pop_model_transform = get_population_model(
        args.num_variables, root=".datasets", download=True
    )
    classifier = get_network(
        args.num_variables,
        depth=args.depth,
        size=args.size,
        root=".datasets",
        download=True,
    )

    var_values = {}
    for var in input_space.attribute_names:
        match input_space.attribute_type(var):
            case TabularInputSpace.AttributeType.INTEGER:
                min_val, max_val = input_space.attribute_bounds(var)
                var_values[var] = torch.tensor(
                    [[i] for i in range(int(min_val), int(max_val) + 1)]
                )
            case TabularInputSpace.AttributeType.CATEGORICAL:
                num_vals = len(input_space.attribute_values(var))
                var_values[var] = torch.tensor(
                    [[float(i == j) for i in range(num_vals)] for j in range(num_vals)]
                )
            case _:
                raise ValueError()

    male_i = input_space.encoding_layout["SEX"]["Male"]
    female_i = input_space.encoding_layout["SEX"]["Female"]

    timeout = args.timeout
    if timeout is None:
        timeout = float("inf")
    max_batch_size = 2**16

    start_time = time()

    # We fuse the first 3 variables into one batch and iterate over the values of
    # the remaining variables
    print("Build Base Batch.")
    fused_vars = 3
    fused = dataset.variables[:fused_vars]
    iterate = dataset.variables[fused_vars:]
    base_batch = list(itertools.product(*[var_values[var] for var in fused]))
    base_batch = [torch.cat(vals) for vals in base_batch]
    base_batch = torch.stack(base_batch, dim=0)

    p_female_lb = 0.0
    p_male_lb = 0.0
    p_high_income_female_lb = 0.0
    p_high_income_female_ub = 1.0
    p_high_income_male_lb = 0.0
    p_high_income_male_ub = 1.0

    print("Iterating over Remaining Values.")
    combinations = itertools.product(*[var_values[var] for var in iterate])
    verification_status = None
    probability_bounds = None
    while (time() - start_time) < timeout:
        other_vals = next(combinations)
        x = torch.concat(
            [base_batch]
            + [
                val.unsqueeze(0).expand(len(base_batch), *val.shape)
                for val in other_vals
            ],
            dim=1,
        )

        z = pop_model_transform(x)
        y = classifier(z)
        probs = input_distribution.probability((x, x))

        p_female_lb += torch.sum((x[:, female_i] >= 1.0).float() * probs).item()
        p_male_lb += torch.sum((x[:, male_i] >= 1.0).float() * probs).item()
        p_high_income_female_lb += torch.sum(
            ((x[:, female_i] >= 1.0) & (y[:, 1] >= y[:, 0])).float() * probs
        ).item()
        p_high_income_female_ub -= torch.sum(
            ((x[:, female_i] <= 0.0) | (y[:, 1] < y[:, 0])).float() * probs
        ).item()
        p_high_income_male_lb += torch.sum(
            ((x[:, male_i] >= 1.0) & (y[:, 1] >= y[:, 0])).float() * probs
        ).item()
        p_high_income_male_ub -= torch.sum(
            ((x[:, male_i] <= 0.0) | (y[:, 1] < y[:, 0])).float() * probs
        ).item()

        p_high_given_female_lb = p_high_income_female_lb / (1 - p_male_lb)
        p_high_given_female_ub = p_high_income_female_ub / (1 - p_female_lb)
        p_high_given_male_lb = p_high_income_male_lb / (1 - p_female_lb)
        p_high_given_male_ub = p_high_income_male_ub / (1 - p_male_lb)
        try:
            lb = p_high_given_female_lb / p_high_given_male_ub
        except ZeroDivisionError:
            lb = -float("inf")
        try:
            ub = p_high_given_female_ub / p_high_given_male_lb
        except ZeroDivisionError:
            ub = float("inf")
        probability_bounds = {
            "P(high income | female)": (
                p_high_given_female_lb,
                p_high_income_female_ub,
            ),
            "P(high income | male)": (p_high_given_male_lb, p_high_income_male_ub),
        }
        if args.log:
            print(
                f"{p_high_given_female_lb:.4f}"
                f" <= P(high income | female) <= "
                f"{p_high_given_female_ub:.4f}"
            )
            print(
                f"{p_high_given_male_lb:.4f}"
                f" <= P(high income | male) <= "
                f"{p_high_given_male_ub:.4f}"
            )
            print(f"{lb:.4f} <= ratio <= {ub:.4f}")
        if lb >= 1 - args.fairness_eps:
            verification_status = VerifyStatus.SATISFIED
            break
        elif ub < 1 - args.fairness_eps:
            verification_status = VerifyStatus.VIOLATED
            break
    end_time = time()

    print(verification_status)
    print(probability_bounds)
    print(f"Runtime: {end_time - start_time:.4f}s")
