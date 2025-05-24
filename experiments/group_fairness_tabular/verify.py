# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import argparse
from pathlib import Path
from time import time

import torch

from experiments.utils import log_machine_and_code_details
from probspecs import (
    Verifier,
    prob,
    compose,
    ExternalFunction,
    ExternalVariable,
    or_expr,
)
from probspecs.utils.yaml import yaml
from experiments.input_spaces import adult_input_space

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Verify Group Fairness")
    parser.add_argument("-d", "--dataset", choices=("Adult",), required=True)
    parser.add_argument(
        "-p",
        "--population-model",
        choices=("independent", "bayesian-network"),
        required=True,
    )
    parser.add_argument(
        "-c",
        "--fairness-criterion",
        choices=("demographic-parity", "parity-of-qualified-individuals"),
        required=True,
    )
    groups = (
        "Male",
        "Female",
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Race-Other",
        "Black",
        "Non-White",
        "Married",
        "Own-child",
        "Unmarried",
    )
    parser.add_argument(
        "-ag",
        "--advantaged-group",
        choices=groups,
        help="The advantaged subpopulation.",
    )
    parser.add_argument(
        "-dg",
        "--disadvantaged-group",
        choices=groups,
        help="The disadvantaged subpopulation.",
    )
    parser.add_argument("-n", "--network", default="tiny_network.pyt")
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
        "--log", action="store_true", help="Whether to print progress messages."
    )
    args = parser.parse_args()

    print("Running Experiment: Group Fairness")
    print("=" * 100)
    print("Command Line Arguments:")
    print(args)
    log_machine_and_code_details()

    match args.dataset:
        case "Adult":
            resource_dir = Path("resources/adult")
        case _:
            raise ValueError()

    population_model = args.population_model.replace("-", "_")
    input_distribution, input_space, pop_model_transform = torch.load(
        resource_dir / f"{population_model}_population_model.pyt"
    )
    classifier = torch.load(resource_dir / args.network)

    x = ExternalVariable("x")
    classifier_func = ExternalFunction("classifier", ("z",))
    pop_transform_func = ExternalFunction("pop_transform", ("x",))
    models_composed = compose(classifier_func, z=pop_transform_func)

    networks = {"classifier": classifier, "pop_transform": pop_model_transform}

    if args.dataset == "Adult":

        def get_group_indicator(group):
            match group:
                case "Female" | "Male":
                    value_i = adult_input_space.encoding_layout["sex"][group]
                    return pop_transform_func[:, value_i] >= 1.0
                case "White" | "Asian-Pac-Islander" | "Amer-Indian-Eskimo" | "Black":
                    value_i = adult_input_space.encoding_layout["race"][group]
                    return pop_transform_func[:, value_i] >= 1.0
                case "Race-Other":
                    value_i = adult_input_space.encoding_layout["race"]["Other"]
                    return pop_transform_func[:, value_i] >= 1.0
                case "Own-Child" | "Unmarried" | "Wife" | "Husband":
                    value_i = adult_input_space.encoding_layout["relationship"][group]
                    return pop_transform_func[:, value_i] >= 1.0
                case "Non-White":
                    return or_expr(
                        *(
                            get_group_indicator(g)
                            for g in (
                                "Asian-Pac-Islander",
                                "Amer-Indian-Eskimo",
                                "Race-Other",
                                "Black",
                            )
                        )
                    )
                case "Married":
                    return or_expr(
                        *(get_group_indicator(g) for g in ("Wife", "Husband"))
                    )

        age_i = adult_input_space.encoding_layout["age"]

        high_income = models_composed[:, 0] < models_composed[:, 1]
        qualified = x[:, age_i] >= 18.0
        disadvantaged = get_group_indicator(args.disadvantaged_group)
        advantaged = get_group_indicator(args.advantaged_group)
        good_outcome = high_income
    else:
        raise NotImplementedError()

    match args.fairness_criterion:
        case "parity-of-qualified-individuals":
            disadvantaged = disadvantaged & qualified
            advantaged = advantaged & qualified

    p_disadvantaged = prob(good_outcome, condition=disadvantaged)
    p_advantaged = prob(good_outcome, condition=advantaged)
    is_fair = p_disadvantaged / p_advantaged > 1 - args.fairness_eps

    if "{" in args.probability_bounds_config or "\n" in args.probability_bounds_config:
        prob_bounds_config = args.probability_bounds_config
    else:
        prob_bounds_config = Path(args.probability_bounds_config)
    prob_bounds_config = yaml.load(prob_bounds_config)
    prob_bounds_config = {"batch_size": 512, "log": args.log} | prob_bounds_config
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
