# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import argparse
from pathlib import Path
from time import time

import torch
from miniacsincome import get_network, get_population_model

from experiments.utils import log_machine_and_code_details
from probspecs import (
    Verifier,
    prob,
    compose,
    ExternalFunction,
    ExternalVariable,
)
from probspecs.utils.yaml import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Verify Group Fairness")
    parser.add_argument(
        "-v",
        "--num-variables",
        choices=tuple(range(1, 9)),
        type=int,
        required=True,
        help="The number of input variables. "
        "Determines the complexity of the verification problem.",
    )
    parser.add_argument("-d", "--depth", default=None)
    parser.add_argument("-s", "--size", default=None)
    parser.add_argument("--fairness-eps", default=0.2)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="A timeout for computing bounds on the frequency of property violations "
        "in seconds.",
    )
    parser.add_argument(
        "--probability-bounds-config",
        default="{}",
        help="A configuration for computing bounds. Can be a path to a YAML file "
        "or a yaml string. Have a look at the ProbabilityBounds class for details "
        "on which configurations are available.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="The random seed for the probability bounds heuristics that use randomness.",
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether to print progress messages."
    )
    args = parser.parse_args()

    print("Running Experiment: MiniACSIncome - Verify Group Fairness")
    print("=" * 100)
    print("Command Line Arguments:")
    print(args)
    log_machine_and_code_details()

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

    male_i = input_space.encoding_layout["SEX"]["Male"]
    female_i = input_space.encoding_layout["SEX"]["Female"]

    # statistically evaluate the network under the population model
    # FIXME: this causes some problem with auto_LiRPA as it seems.
    # n = 10000
    # inputs = input_distribution.sample(n)
    # inputs = pop_model_transform(inputs.float())
    # male_inputs = inputs[inputs[:, male_i] == 1.0]
    # female_inputs = inputs[inputs[:, female_i] == 1.0]
    # pred_male = torch.argmax(classifier(male_inputs), dim=-1)
    # pred_female = torch.argmax(classifier(female_inputs), dim=-1)
    # print("=" * 100)
    # print(
    #     f"Distribution Male/Female: "
    #     f"{100*len(male_inputs)/len(inputs):3.1f}%/"
    #     f"{100*len(female_inputs)/len(inputs):3.1f}%\n"
    #     f"Class 1 Frequency (Male/Female): "
    #     f"{100 * (pred_male == 0).float().mean():3.1f}%/"
    #     f"{100 * (pred_female == 0).float().mean():3.1f}%\n"
    #     f"Class 2 Frequency (Male/Female): "
    #     f"{100 * (pred_male == 1).float().mean():3.1f}%/"
    #     f"{100 * (pred_female == 1).float().mean():3.1f}%"
    # )
    # print("=" * 100)
    # sys.exit()

    x = ExternalVariable("x")
    classifier_func = ExternalFunction("classifier", ("z",))
    pop_transform_func = ExternalFunction("pop_transform", ("x",))
    models_composed = compose(classifier_func, z=pop_transform_func)

    networks = {"classifier": classifier, "pop_transform": pop_model_transform}

    male = pop_transform_func[:, male_i] >= 1.0
    female = pop_transform_func[:, female_i] >= 1.0
    high_income = models_composed[:, 0] < models_composed[:, 1]

    p_female = prob(high_income, condition=female)
    p_male = prob(high_income, condition=male)
    is_fair = p_female / p_male > 1 - args.fairness_eps

    if "{" in args.probability_bounds_config or "\n" in args.probability_bounds_config:
        prob_bounds_config = args.probability_bounds_config
    else:
        prob_bounds_config = Path(args.probability_bounds_config)
    prob_bounds_config = yaml.load(prob_bounds_config)

    prob_bounds_config = {
        "batch_size": 512,
        "random_seed": args.random_seed,
        "log": args.log,
    } | prob_bounds_config
    verifier = Verifier(
        worker_devices="cpu",
        timeout=args.timeout,
        log=args.log,
        probability_bounds_config=prob_bounds_config,
    )

    print("Starting Verification")
    start_time = time()
    verification_status, probability_bounds = verifier.verify(
        is_fair,
        networks,
        {"x": input_space},
        {"x": input_distribution},
    )
    end_time = time()
    print(verification_status)
    print(probability_bounds)
    print(f"Runtime: {end_time - start_time:.4f}s")
