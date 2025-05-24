# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import argparse
from pathlib import Path
from time import time

from experiments.utils import log_machine_and_code_details
from probspecs import Verifier, prob, compose, ExternalFunction, ExternalVariable
from experiments.fairsquare.population_models import *
from experiments.fairsquare.classifiers import *
from probspecs.bounds import WarmStartBounds, SwitchToHeuristicsWithGuarantees
from probspecs.utils.yaml import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Verify FairSquare Models")
    parser.add_argument(
        "-p",
        "--population-model",
        choices=("rind", "uind", "eBN", "eBNc", "rBN", "rBNc", "BN", "BNc", "BNcc"),
        required=True,
        help="The population model. "
        "rind: independent with realistic bounds (age < 0 disallowed), "
        "uind: independent with unrealistic bounds (age < 0 allowed), "
        "eBN: the FairSquare Bayesian network, encoded as a probability distribution, "
        "with unrealistic bounds (age < 0 allowed), "
        "eBNc: eBN with an integrity constraint."
        "rBN: the FairSquare Bayesian network, encoded as a probability distribution, "
        "with realistic bounds (age < 0 disallowed), "
        "rBNc: rBN with an integrity constraint."
        "BN: Bayesian network encoded as independent distributions "
        "and an input transformation, as in FairSquare, "
        "BNc: BN with integrity constraint"
        "BNcc: BNc with additional clipping, ",
    )
    parser.add_argument(
        "-c", "--classifier", choices=("NN_V2H1", "NN_V2H2", "NN_V3H2"), required=True
    )
    qual_group = parser.add_mutually_exclusive_group(required=True)
    qual_group.add_argument("-q", "--qual", action="store_true")
    qual_group.add_argument("-n", "--no-qual", action="store_true")
    parser.add_argument("--fairness-eps", default=0.15)
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

    print("Running Experiment: FairSquare")
    print("=" * 100)
    print("Command Line Arguments:")
    print(args)
    log_machine_and_code_details()

    match args.population_model:
        case "rind":
            pop_model = IndependentPopulationModel()
        case "uind":
            pop_model = IndependentPopulationModel(realistic=False)
        case "BN":
            pop_model = BayesianNetworkPopulationModel()
        case "BNc":
            pop_model = BayesianNetworkPopulationModel(integrity_constraint=True)
        case "BNcc":
            pop_model = BayesianNetworkPopulationModel(True, True)
        case "eBN":
            pop_model = ExplicitBayesianNetworkPopulationModel(realistic=False)
        case "eBNc":
            pop_model = ExplicitBayesianNetworkPopulationModel(
                realistic=False, integrity_constraint=True
            )
        case "rBN":
            pop_model = ExplicitBayesianNetworkPopulationModel(realistic=True)
        case "rBNc":
            pop_model = ExplicitBayesianNetworkPopulationModel(
                realistic=True, integrity_constraint=True
            )
        case _:
            raise ValueError()

    match args.classifier:
        case "NN_V2H1":
            classifier = FairSquareNNV2H1()
        case "NN_V2H2":
            classifier = FairSquareNNV2H2()
        case "NN_V3H2":
            classifier = FairSquareNNV3H2()
        case _:
            raise ValueError()

    x = ExternalVariable("x")
    classifier_func = ExternalFunction("classifier", ("z",))
    networks = {"classifier": classifier}
    if pop_model.population_model is not None:
        pop_model_func = ExternalFunction("pop_model", ("x",))
        models_composed = compose(classifier_func, z=pop_model_func)
        networks["pop_model"] = pop_model.population_model
    else:
        pop_model_func = x
        models_composed = compose(classifier_func, z=x)

    input_lbs, input_ubs = pop_model.input_space.input_bounds
    valid_input = (x[:, 0] >= input_lbs[0]) & (x[:, 0] <= input_ubs[0])
    for i in range(1, input_lbs.size(0)):
        valid_input = (
            valid_input & (x[:, i] >= input_lbs[i]) & (x[:, i] <= input_ubs[i])
        )

    female = x[:, 2] <= 0.0
    male = x[:, 2] >= 1.0
    high_income = models_composed[:, 0] < models_composed[:, 1]
    base_cond = valid_input
    if args.qual:
        qualified = pop_model_func[:, 0] >= 18.0
        base_cond = base_cond & qualified
    p_disadvantaged = prob(high_income, condition=female & base_cond)
    p_advantaged = prob(high_income, condition=male & base_cond)
    is_fair = p_disadvantaged / p_advantaged > 1 - args.fairness_eps

    if "{" in args.probability_bounds_config or "\n" in args.probability_bounds_config:
        prob_bounds_config = args.probability_bounds_config
    else:
        prob_bounds_config = Path(args.probability_bounds_config)
    prob_bounds_config = yaml.load(prob_bounds_config)
    prob_bounds_config = {"batch_size": 1024, "log": args.log} | prob_bounds_config
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
        {"x": pop_model.input_space},
        {"x": pop_model.probability_distribution},
    )
    end_time = time()
    print(verification_status)
    print(probability_bounds)
    print(f"Runtime: {end_time - start_time:.4f}s")
