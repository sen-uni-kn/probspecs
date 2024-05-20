#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import argparse
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
        rt = pd.to_numeric(rt[rt != "TO"])
        rt.sort_values(ascending=True, inplace=True)
        rt = pd.DataFrame({"Nr": range(1, len(rt) + 1), "Runtime": rt})
        out_file = experiment_directory / "Runtimes.csv"
        print("Writing", out_file)
        rt.to_csv(out_file)
