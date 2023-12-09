# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import argparse

from probspecs import verify, prob, compose, ExternalFunction, ExternalVariable
from experiments.fairsquare.population_models import *
from experiments.fairsquare.classifiers import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Verify FairSquare Models")
    parser.add_argument(
        "-p", "--population-model", choices=("ind", "BN", "BNc"), required=True
    )
    parser.add_argument(
        "-c", "--classifier", choices=("NN_V2H1", "NN_V2H2", "NN_V3H2"), required=True
    )
    qual_group = parser.add_mutually_exclusive_group(required=True)
    qual_group.add_argument("-q", "--qual", action="store_true")
    qual_group.add_argument("-n", "--no-qual", action="store_true")
    parser.add_argument("--fairness-eps", default=0.15)
    args = parser.parse_args()

    match args.population_model:
        case "ind":
            pop_model = IndependentPopulationModel()
        case "BN":
            raise NotImplementedError()
        case "BNc":
            raise NotImplementedError()
        case _:
            raise ValueError()

    match args.classifier:
        case "NN_V2H1":
            classifier = FairSquareNNV2H1()
        case "NN_V2H2":
            raise NotImplementedError()
        case "NN_V3H2":
            raise NotImplementedError()
        case _:
            raise ValueError()

    x = ExternalVariable("x")
    classifier_func = ExternalFunction("classifier", ("x",))
    networks = {"classifier": classifier}
    if pop_model.population_model is not None:
        pop_model_func = ExternalFunction("pop_model", ("x",))
        models_composed = compose(classifier_func, z=pop_model_func)
        networks["pop_model"] = pop_model.population_model
    else:
        models_composed = classifier_func

    input_lbs, input_ubs = pop_model.input_space.input_bounds
    valid_input = (x[:, 0] >= input_lbs[0]) & (x[:, 0] <= input_ubs[0])
    for i in range(1, input_lbs.size(0)):
        valid_input = (
            valid_input & (x[:, i] >= input_lbs[i]) & (x[:, i] <= input_ubs[i])
        )

    female = x[:, 2] <= 0
    male = x[:, 2] >= 1
    high_income = models_composed[:, 0] < models_composed[:, 1]
    base_cond = valid_input
    if args.qual:
        qualified = x[:, 0] >= 18
        base_cond = base_cond & qualified
    p_disadvantaged = prob(high_income, condition=female & base_cond)
    p_advantaged = prob(high_income, condition=male & base_cond)
    is_fair = p_disadvantaged / p_advantaged > 1 - args.fairness_eps

    verification_status, probability_bounds = verify(
        is_fair,
        networks,
        {"x": pop_model.input_space},
        {"x": pop_model.probability_distribution},
        split_heuristic="IBP",
    )
    print(verification_status)
    print(probability_bounds)
