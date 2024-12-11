#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compile Random Branch and Split Selection")
    parser.add_argument(
        "directories",
        nargs="+",
        type=str,
        help="The directories containing the experiment results. "
        "For example experiments/output/SOME_TIMESTAMP/",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Where to store the resulting aggregate runtime CSV file.",
    )
    args = parser.parse_args()
    experiment_directories = args.directories

    runtimes = defaultdict(list)
    for experiment_directory in experiment_directories:
        experiment_directory = Path(experiment_directory)
        mini_acs_income_df = pd.read_csv(
            experiment_directory / "mini_acs_income" / "verify" / "results.csv"
        )
        mini_acs_income_df.columns = mini_acs_income_df.columns.str.replace(" ", "")
        mini_acs_income_df.replace("TO", 3600, inplace=True)

        for _, row in mini_acs_income_df.iterrows():
            runtimes[row["InputVariables"]].append(float(row["Runtime"]))

    median_runtime = {
        in_vars: pd.Series(rts).median() for in_vars, rts in runtimes.items()
    }
    min_runtime = {in_vars: pd.Series(rts).min() for in_vars, rts in runtimes.items()}
    max_runtime = {in_vars: pd.Series(rts).max() for in_vars, rts in runtimes.items()}
    aggregate_df = pd.DataFrame(
        [
            {
                "InputVariables": in_vars,
                "MinRuntime": min_runtime[in_vars],
                "MedianRuntime": median_runtime[in_vars],
                "MaxRuntime": max_runtime[in_vars],
            }
            for in_vars in median_runtime
        ]
    )
    aggregate_df.sort_values(by="InputVariables", ascending=True, inplace=True)

    out_file = Path(args.out)
    print("Writing aggregate runtimes to", out_file)
    aggregate_df.to_csv(out_file, index=False)
