# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import argparse
from pathlib import Path
from time import time

import dill
import torch

from probspecs import verify, prob, compose, ExternalFunction, ExternalVariable
from experiments.group_fairness_tabular.input_spaces import adult_input_space

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Verify Group Fairness")
    parser.add_argument("-d", "--dataset", choices=("Adult",), required=True)
    parser.add_argument(
        "-p",
        "--population-model",
        choices=("independent", "bayesian-network", "neural-network"),
        required=True,
    )
    parser.add_argument(
        "-c",
        "--fairness-criterion",
        choices=("demographic-parity", "parity-of-qualified-individuals"),
        required=True,
    )
    parser.add_argument("-n", "--network", default="network.pyt")
    parser.add_argument("--fairness-eps", default=0.2)
    args = parser.parse_args()

    match args.dataset:
        case "Adult":
            resource_dir = Path("resources/adult")
        case _:
            raise ValueError()

    match args.population_model:
        case "independent":
            input_distribution, input_space, pop_model_transform = torch.load(
                resource_dir / "independent_population_model.pyt"
            )
        case "factor_analysis":
            with open(resource_dir / "factor_analysis_input_space.dill", "rb") as file:
                input_space = dill.load(file)
            with open(resource_dir / "factor_analysis_distribution.dill", "rb") as file:
                input_distribution = dill.load(file)
            pop_model_transform = torch.load(
                resource_dir / "factor_analysis_population_model.pyt"
            )
        case _:
            raise ValueError()

    classifier = torch.load(resource_dir / args.network)

    x = ExternalVariable("x")
    classifier_func = ExternalFunction("classifier", ("z",))
    pop_transform_func = ExternalFunction("pop_transform", ("x",))
    models_composed = compose(classifier_func, z=pop_transform_func)

    networks = {"classifier": classifier, "pop_transform": pop_model_transform}

    if args.dataset == "Adult":
        female_i = adult_input_space.encoding_layout["sex"]["Female"]
        male_i = adult_input_space.encoding_layout["sex"]["Male"]
        age_i = adult_input_space.encoding_layout["age"]

        female = pop_transform_func[:, female_i] >= 1.0
        male = pop_transform_func[:, male_i] >= 1.0
        high_income = models_composed[:, 0] < models_composed[:, 1]
        qualified = x[:, age_i] >= 18.0

        disadvantaged = female
        advantaged = male
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

    start_time = time()
    verification_status, probability_bounds = verify(
        is_fair,
        networks,
        {"x": input_space},
        {"x": input_distribution},
        batch_size=512,
        split_heuristic="IBP",
        worker_devices=("cpu", "cpu"),
    )
    end_time = time()
    print(verification_status)
    print(probability_bounds)
    print("Runtime: ", end_time - start_time)
