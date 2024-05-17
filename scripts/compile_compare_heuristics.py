#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import argparse
import itertools
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compile Heuristics Comparison")
    parser.add_argument(
        "directories",
        nargs="+",
        type=str,
        help="The directories containing the experiment results. "
        "For example experiments/output/SOME_TIMESTAMP/",
    )
    args = parser.parse_args()
    experiment_directories = args.directories

    runtimes = {}

    for experiment_directory in experiment_directories:
        experiment_directory = Path(experiment_directory)
        fairsquare_df = pd.read_csv(experiment_directory / "fairsquare" / "results.csv")
        robustness_df = pd.read_csv(
            experiment_directory / "acasxu" / "robustness" / "results.csv"
        )
        mini_acs_income_df = pd.read_csv(
            experiment_directory / "mini_acs_income" / "verify" / "results.csv"
        )
        rt = pd.concat(
            [
                fairsquare_df["Runtime"],
                robustness_df["Runtime"],
                mini_acs_income_df["Runtime"],
            ]
        )
        rt.replace("TO", 3600, inplace=True)
        experiment_name = experiment_directory.name
        runtimes[experiment_name] = rt

    runtimes = pd.DataFrame(runtimes)
    out_file = "HeuristicsComparison.csv"
    print("Writing", out_file)
    runtimes.to_csv(out_file, index=False)
