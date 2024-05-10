#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import argparse
import itertools
from pathlib import Path

import pandas as pd


def decimals_only(f):
    if f == 0:  # negative zero to zero
        f = 0.0
    if f >= 0.995:
        f = 0.990
    if f >= 1.0:
        return "1.0"
    else:
        return f".{f*100:02.0f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compile Experiment Results")
    parser.add_argument(
        "directory",
        type=str,
        help="The directory containing the experiment results. "
        "For example experiments/output/SOME_TIMESTAMP/",
    )
    args = parser.parse_args()
    experiment_directory = Path(args.directory)

    # FairSquare Table
    fairsquare_subdir = experiment_directory / "fairsquare"
    fairsquare_df = pd.read_csv(fairsquare_subdir / "results.csv")
    # reorder columns
    fairsquare_df = fairsquare_df[
        ["Qualified", "Network", "Population Model", "Runtime", "Fair"]
    ]
    fairsquare_df.sort_values(
        by=["Qualified", "Network", "Population Model"],
        key=lambda series: series.replace(
            {
                "true": True,
                "false": False,
                "V2H1": 1,
                "V2H2": 2,
                "V3H2": 3,
                "uind": 1,
                "eBN": 2,
                "eBNc": 3,
                "rind": 4,
                "rBN": 5,
                "rBNc": 6,
            }
        ),
        ascending=True,
        inplace=True,
    )
    out_file = experiment_directory / "FairSquareResults.tex"
    print("Writing", out_file)
    network_name_lookup = {
        "V2H1": r"NN\textsubscript{2,1}",
        "V2H2": r"NN\textsubscript{2,2}",
        "V3H2": r"NN\textsubscript{3,2}",
    }
    pop_model_name_lookup = {
        "uind": "independent        ",
        "eBN": "Bayes Net 1        ",
        "eBNc": "Bayes Net 2        ",
        "ind": "independent clipped",
        "rind": "independent clipped",
        "rBN": "Bayes Net 1 clipped",
        "rBNc": "Bayes Net 2 clipped",
    }
    fairsquare_df.to_latex(
        out_file,
        columns=["Qualified", "Network", "Population Model", "Runtime", "Fair"],
        header=True,
        index=False,
        formatters={
            "Qualified": lambda q: "yes" if q else " no",
            "Network": lambda n: network_name_lookup[n],
            "Population Model": lambda p: pop_model_name_lookup[p],
            "Fair": lambda f: "\exsuccess{}" if f else "\exfailure{}",
            "Runtime": lambda r: f"{float(r):6.1f}" if r != "TO" else "    TO",
        },
    )

    # ACAS Xu Safety Tables
    for suffix, out_suffix in (("", ""), ("_less_precise", "LessPrecise")):
        safety_subdir = experiment_directory / "acasxu" / ("safety" + suffix)
        if not safety_subdir.exists():
            continue
        safety_df = pd.read_csv(safety_subdir / "results.csv")
        safety_df.sort_values(
            by=["Property", "Network"],
            key=lambda series: series.replace(
                {
                    f"{i1}_{i2}": i1 * 10 + i2
                    for i1 in range(1, 6)
                    for i2 in range(1, 10)
                }
            ),
            ascending=True,
            inplace=True,
        )
        safety_df["Precision"] = safety_df["Upper Bound"] - safety_df["Lower Bound"]
        # reorder columns
        safety_df = safety_df[
            [
                "Property",
                "Network",
                "Lower Bound",
                "Upper Bound",
                "Precision",
                "Runtime",
                "Abort Reason",
            ]
        ]
        out_file = experiment_directory / f"ACASXuSafety{out_suffix}Results.tex"
        print("Writing", out_file)
        network_name_lookup = {
            f"{i1}_{i2}": f"$N_{{{i1},{i2}}}$"
            for i1 in range(1, 6)
            for i2 in range(1, 10)
        }
        property_lookup = {i: rf"$\varphi_{{{i}}}$" for i in range(1, 11)}
        safety_df.to_latex(
            out_file,
            columns=[
                "Property",
                "Network",
                "Lower Bound",
                "Upper Bound",
                "Precision",
                "Runtime",
            ],
            header=True,
            index=False,
            formatters={
                "Property": lambda p: property_lookup[p],
                "Network": lambda n: network_name_lookup[n],
                "Lower Bound": lambda v: f"{v:6.4f}",
                "Upper Bound": lambda v: f"{v:6.4f}",
                "Precision": lambda v: f"{v:6.4f}",
                "Runtime": lambda r: f"{float(r):6.1f}" if r < 900.0 else "    TO",
            },
        )

    # ACAS Xu Robustness Tables
    robustness_subdir = experiment_directory / "acasxu" / "robustness"
    robustness_df = pd.read_csv(robustness_subdir / "results.csv")

    runtime_summary = robustness_df[["Runtime"]].describe().transpose()
    runtime_summary.columns = [
        "count",
        "mean",
        "std",
        "min",
        r"25\%",
        "median",
        r"75\%",
        "max",
    ]
    out_file = experiment_directory / f"ACASXuRobustnessRuntimeSummary.tex"
    print("Writing", out_file)
    runtime_summary.to_latex(out_file, header=True, index=False, float_format="%6.1f")

    robustness_df.sort_values(
        by=["Network", "Source Label", "Input", "Target Label"],
        key=lambda series: series.replace(
            {f"{i1}_{i2}": i1 * 10 + i2 for i1 in range(1, 6) for i2 in range(1, 10)}
        ),
        ascending=True,
        inplace=True,
    )
    label_replace = {0: "COC", 1: "WL ", 2: "WR ", 3: "SL ", 4: "SR "}
    robustness_df["Source Label"].replace(label_replace, inplace=True)
    robustness_df["Target Label"].replace(label_replace, inplace=True)
    robustness_df["LB"] = robustness_df["Lower Bound"].map(decimals_only)
    robustness_df["UB"] = robustness_df["Upper Bound"].map(decimals_only)
    robustness_df["Bounds"] = robustness_df["LB"] + ", " + robustness_df["UB"]
    robustness_df["Precision"] = (
        robustness_df["Upper Bound"] - robustness_df["Lower Bound"] + 0.0
    )
    out_df = robustness_df.pivot_table(
        index=["Network", "Source Label", "Input"],
        columns="Target Label",
        values=["Bounds", "Precision", "Runtime"],
        aggfunc="first",
    ).reset_index()
    out_df.columns = out_df.columns.reorder_levels(["Target Label", None])
    out_df = out_df[
        [("", "Network"), ("", "Source Label"), ("", "Input")]
        + list(
            itertools.chain(
                *(
                    [(label, "Bounds"), (label, "Precision"), (label, "Runtime")]
                    for label in label_replace.values()
                )
            )
        )
    ]

    out_file = experiment_directory / f"ACASXuRobustnessResults.tex"
    print("Writing", out_file)
    out_df.to_latex(
        out_file,
        columns=[
            ("", "Source Label"),
            ("", "Input"),
        ]
        + list(itertools.product(label_replace.values(), ("Bounds", "Runtime"))),
        header=True,
        index=False,
        formatters={(i, "Bounds"): lambda s: f"${s}$" for i in label_replace.values()}
        | {(i, "Precision"): lambda v: f"{v:6.4f}" for i in label_replace.values()}
        | {
            (i, "Runtime"): lambda r: f"{float(r):6.1f}" if r < 900.0 else "    TO"
            for i in label_replace.values()
        },
    )

    # MiniACSIncome Figure (CSV Export)
    mini_acs_income_subdir = experiment_directory / "mini_acs_income"

    verify_df = pd.read_csv(mini_acs_income_subdir / "verify" / "results.csv")
    verify_df.columns = ["InputVariables", "VerifyFair", "VerifyRuntime"]
    verify_df.index = verify_df["InputVariables"]

    enumerate_df = pd.read_csv(mini_acs_income_subdir / "enumerate" / "results.csv")
    enumerate_df.columns = ["InputVariables", "EnumerateFair", "EnumerateRuntime"]
    enumerate_df.index = enumerate_df["InputVariables"]
    enumerate_df.drop("InputVariables", axis=1, inplace=True)

    df = pd.concat([verify_df, enumerate_df], axis=1)
    df.replace("TO", 3600, inplace=True)
    df.sort_index(inplace=True)
    out_file = experiment_directory / "MiniACSIncomeResults.csv"
    print("Writing", out_file)
    df.to_csv(out_file, index=False)
