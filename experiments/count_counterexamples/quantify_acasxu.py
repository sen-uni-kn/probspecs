#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import argparse
from time import time
from pathlib import Path

from probspecs import (
    prob,
    ExternalFunction,
    ExternalVariable,
    TensorInputSpace,
    Formula,
)
from probspecs.bounds import ProbabilityBounds
from probspecs.distributions import Uniform
from experiments.utils import load_nnet


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Quantify Violations - ACAS Xu")
    parser.add_argument(
        "-n",
        "--network",
        choices=("1_9", "5_3")
        + tuple(f"2_{i}" for i in range(10))
        + (
            "net_1_9_property_7_partially_repaired_1",
            "net_1_9_property_7_partially_repaired_2",
            "net_1_9_property_7_partially_repaired_3",
            "net_2_9_property_8_unknown",
        ),
        required=True,
        help="The ACAS Xu network number of the network to investigate. "
        "The network is loaded from resouces/acasxu.",
    )
    parser.add_argument(
        "-p",
        "--property",
        choices=(2, 7, 8),
        type=int,
        required=True,
        help="The ACAS Xu property to investigate.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="A timeout for computing bounds on the frequency of property violations "
        "in seconds.",
    )
    args = parser.parse_args()

    net_split = args.network.split("_")
    if len(net_split) == 2:
        net_i1, net_i2 = net_split
        net_name = f"ACASXU_run2a_{net_i1}_{net_i2}_batch_2000"
    else:
        net_name = args.network
    network, (input_lbs, input_ubs) = load_nnet(
        Path("resources", "acasxu", net_name + ".nnet")
    )
    input_space = TensorInputSpace(input_lbs, input_ubs)

    x = ExternalVariable("x")
    # x[:, 0]: rho (distance between ownship and intruder)
    # x[:, 1]: theta (angle between ownship heading angle and intruder)
    # x[:, 2]: psi (angle between ownship heading angle and intruder heading angle)
    # x[:, 3]: v_own (ownship speed)
    # x[:, 4]: v_int (intruder speed)
    net_func = ExternalFunction("network", ("x",))
    # net_func[:, 0]: score for Clear-of-Conflict (CoC)
    # net_func[:, 1]: score for weak left
    # net_func[:, 2]: score for weak right
    # net_func[:, 3]: score for strong left
    # net_func[:, 4]: score for strong right
    # Classifier chooses the class with the *minimal* score.
    input_constraint = None
    output_constraint = None
    pi = 3.141592
    match args.property:
        case 2:
            # Intruder distant and significantly slower than the ownship
            # => Clear of Conflict (first) score may not be maximal
            input_constraint = (
                (x[:, 0] >= 55947.691) & (x[:, 3] >= 1145) & (x[:, 4] <= 60)
            )
            output_constraint = Formula(
                Formula.Operator.OR,
                tuple(net_func[:, 0] < net_func[:, i] for i in range(1, 5)),
            )
        case 7:
            # No strong turns if vertical separation is large
            # strong right is 4th output, strong left is 5th output
            input_constraint = (
                (0.0 <= x[:, 0])
                & (x[:, 0] <= 60760.0)
                & (-pi <= x[:, 1])
                & (x[:, 1] <= pi)
                & (-pi <= x[:, 2])
                & (x[:, 2] <= pi)
                & (100.0 <= x[:, 3])
                & (x[:, 3] <= 1200.0)
                & (0.0 <= x[:, 4])
                & (x[:, 4] <= 1200.0)
            )
            output_constraint = Formula(
                Formula.Operator.OR,
                tuple(net_func[:, 3] > net_func[:, i] for i in range(3)),
            ) & Formula(
                Formula.Operator.OR,
                tuple(net_func[:, 4] > net_func[:, i] for i in range(3)),
            )
        case 8:
            # Large vertical separation and weak left was previously
            # advised (only applies to net 2,9)
            # => output is COC (1st output) or weak left (2nd output)
            input_constraint = (
                (0.0 <= x[:, 0])
                & (x[:, 0] <= 60760.0)
                & (-pi <= x[:, 1])
                & (x[:, 1] <= -0.75 * pi)
                & (-0.1 <= x[:, 2])
                & (x[:, 2] <= 0.1)
                & (600.0 <= x[:, 3])
                & (x[:, 3] <= 1200.0)
                & (600.0 <= x[:, 4])
                & (x[:, 4] <= 1200.0)
            )
            output_constraint = Formula(
                Formula.Operator.AND,
                tuple(net_func[:, 0] < net_func[:, i] for i in range(2, 5)),
            ) | Formula(
                Formula.Operator.AND,
                tuple(net_func[:, 1] < net_func[:, i] for i in range(2, 5)),
            )
    p_violation = prob(~output_constraint, condition=input_constraint)
    # count violations: uniform distribution
    input_distribution = Uniform(input_space.input_bounds)

    timeout = args.timeout
    if timeout is None:
        timeout = float("inf")
    compute_bounds = ProbabilityBounds(batch_size=128, device="cpu")
    start_time = time()
    bounds_gen = compute_bounds.bound(
        p_violation,
        {"network": network},
        {"x": input_space},
        {"x": input_distribution},
    )
    best_bounds = None
    while (time() - start_time) < timeout:
        best_bounds = next(bounds_gen)
        lower, upper = best_bounds
        print(f"{lower:.6f} <= P(violation) <= {upper:.6f}")
    print(f"Finished.")
