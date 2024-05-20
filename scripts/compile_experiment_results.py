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
        f = 1.0
    if f >= 1.0:
        return "1.0"
    else:
        return f".{f*100:02.0f}"


def as_percentage(f):
    if f == 0:  # negative zero to zero
        f = 0.0
    if f >= 1.0:
        return r" 100\%"
    else:
        return rf"{f*100:2.1f}\%"


def bound_formatter(val, dollars=True):
    formatted = rf"{val*100:6.2f}\%"
    if dollars:
        formatted = f"${formatted}$"
    return formatted


def bound_formatter2(val):
    return bound_formatter(val, dollars=False)


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
    is_fair_lookup = {
        "True": r"\exsuccess{}",
        "False": r"\exfailure{}",
        "Unknown": r"\exunknown{}",
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
            "Fair": lambda f: is_fair_lookup[f],
            "Runtime": lambda r: f"{float(r):6.1f}" if r != "TO" else "    TO",
        },
    )

    # ACAS Xu Safety Tables
    for suffix, out_suffix, timeout in ():
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
        property_lookup = {i: rf"$\varphi_{{{i}}}$" for i in range(1, 11)} | {
            f"property{i}": rf"$\varphi_{{{i}}}$" for i in range(1, 11)
        }
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
                "Lower Bound": lambda v: rf"{v*100:6.2f}\%",
                "Upper Bound": lambda v: rf"{v*100:6.2f}\%",
                "Precision": lambda v: rf"{v*100:6.2f}\%",
                "Runtime": lambda r: f"{float(r):6.1f}" if r < timeout else "    TO",
            },
        )

    # ACAS Xu Safety Tables
    safety_subdir = experiment_directory / "acasxu" / "safety"
    dfs = []
    timeouts = []
    for timeout_dir in [d for d in safety_subdir.iterdir() if d.is_dir()]:
        timeout = timeout_dir.name
        timeouts.append(timeout)
        results_df = pd.read_csv(timeout_dir / "results.csv")
        results_df.index = pd.MultiIndex.from_frame(results_df[["Property", "Network"]])
        results_df.drop(columns=["Network", "Property"], inplace=True)
        results_df.drop(columns=["Runtime", "Abort Reason"], inplace=True)
        lb = results_df["Lower Bound"].map(bound_formatter2)
        ub = results_df["Upper Bound"].map(bound_formatter2)
        results_df["Bounds"] = "$" + lb + ", " + ub + "$"
        results_df["Precision"] = results_df["Upper Bound"] - results_df["Lower Bound"]
        results_df.columns = pd.MultiIndex.from_product(
            [(timeout,), ("Lower Bound", "Upper Bound", "Bounds", "Precision")]
        )
        dfs.append(results_df)
    df = dfs[0].join(dfs[1:], how="outer")

    df["Property"] = df.index.get_level_values("Property")
    df["Network"] = df.index.get_level_values("Network")
    df["Instance Name"] = (
        df.index.get_level_values("Property")
        + "_"
        + df.index.get_level_values("Network")
    )

    df.sort_index(inplace=True)
    timeouts.sort(key=lambda x: int(x))
    column_order = [("Property", ""), ("Network", ""), ("Instance Name", "")] + list(
        itertools.chain(
            *[
                [
                    (timeout, "Lower Bound"),
                    (timeout, "Upper Bound"),
                    (timeout, "Bounds"),
                    (timeout, "Precision"),
                ]
                for timeout in timeouts
            ]
        )
    )
    df = df[column_order]

    # Utilities for LaTeX formatting
    network_name_lookup = {
        f"{i1}_{i2}": f"$N_{{{i1},{i2}}}$" for i1 in range(1, 6) for i2 in range(1, 10)
    }
    property_lookup = {i: rf"$\varphi_{{{i}}}$" for i in range(1, 11)} | {
        f"property{i}": rf"$\varphi_{{{i}}}$" for i in range(1, 11)
    }

    # Rename index levels for nicer LaTex export
    df.index = df.index.set_levels(
        [
            [property_lookup[p] for p in df.index.unique("Property").sort_values()],
            [network_name_lookup[n] for n in df.index.unique("Network").sort_values()],
        ],
    )

    # Short Table
    selection = [
        (property_lookup["property2"], network_name_lookup["4_3"]),
        (property_lookup["property2"], network_name_lookup["4_9"]),
        (property_lookup["property2"], network_name_lookup["5_8"]),
        (property_lookup["property7"], network_name_lookup["1_9"]),
        (property_lookup["property8"], network_name_lookup["2_9"]),
    ]
    selection = df[df.index.isin(selection)]
    out_file = experiment_directory / f"ACASXuSafetySelected.tex"
    print("Writing", out_file)
    selection.to_latex(
        out_file,
        columns=list(
            itertools.chain(
                *[[(timeout, "Bounds"), (timeout, "Precision")] for timeout in timeouts]
            )
        ),
        header=True,
        index=True,
        float_format=bound_formatter,
    )

    # Table with more networks but fewer timeouts
    out_file = experiment_directory / f"ACASXuSafetyFull.tex"
    print("Writing", out_file)
    df.to_latex(
        out_file,
        columns=list(
            itertools.chain(
                *[
                    [(timeout, "Bounds"), (timeout, "Precision")]
                    for timeout in ["10", "30", "60"]
                ]
            )
        ),
        header=True,
        index=True,
        float_format=bound_formatter,
        na_rep="--",
    )

    # ACAS Xu Robustness Tables
    robustness_subdir = experiment_directory / "acasxu" / "robustness"
    robustness_df = pd.read_csv(robustness_subdir / "results.csv")
    #

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
        ascending=True,
        inplace=True,
    )
    label_replace = {0: "COC", 1: "WL ", 2: "WR ", 3: "SL ", 4: "SR "}
    # robustness_df["Source Label"].replace(label_replace, inplace=True)
    robustness_df["Target Label"].replace(label_replace, inplace=True)
    robustness_df["Input"] = robustness_df["Input"] + 1  # indexing from 1
    # robustness_df["LB"] = robustness_df["Lower Bound"].map(decimals_only)
    # robustness_df["UB"] = robustness_df["Upper Bound"].map(decimals_only)
    robustness_df["LB"] = robustness_df["Lower Bound"].map(as_percentage)
    robustness_df["UB"] = robustness_df["Upper Bound"].map(as_percentage)
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
    out_df[("Source Label", "")].replace(label_replace, inplace=True)
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
    verify_df.columns = verify_df.columns.str.replace(" ", "")

    enumerate_df = pd.read_csv(mini_acs_income_subdir / "enumerate" / "results.csv")
    enumerate_df.columns = enumerate_df.columns.str.replace(" ", "")
    enumerate_df.rename(
        columns={"Runtime": "EnumerateRuntime", "Fair": "EnumerateFair"}, inplace=True
    )

    # Input Variables Figure
    select = (verify_df["NumNeurons"] == 10) & (verify_df["NumLayers"] == 1)
    small_network_df = verify_df[select].copy()
    small_network_df.rename(
        columns={"Runtime": "VerifyRuntime", "Fair": "VerifyFair"}, inplace=True
    )
    small_network_df.index = small_network_df["InputVariables"]
    enumerate_df.index = enumerate_df["InputVariables"]
    enumerate_df.drop(
        ["InputVariables", "NumNeurons", "NumLayers"], axis=1, inplace=True
    )
    df = pd.concat([small_network_df, enumerate_df], axis=1)
    df.replace("TO", 3600, inplace=True)
    df.sort_index(inplace=True)
    out_file = experiment_directory / "MiniACSIncomeInputSize.csv"
    print("Writing", out_file)
    df.to_csv(out_file, index=False)

    # Network Size Figure
    select = (verify_df["InputVariables"] == 4) & (verify_df["NumLayers"] == 1)
    width = verify_df[select].copy()
    # 4 inputs x size -> size x 2 outputs
    width["NumParameters"] = (4 + 2) * width["NumNeurons"]
    # width.replace("TO", 3600, inplace=True)
    # width.sort_values(by="NumParameters", inplace=True)
    # out_file = experiment_directory / "MiniACSIncomeNumNeurons.csv"
    # print("Writing", out_file)
    # width.to_csv(out_file, index=False)

    select = (verify_df["InputVariables"] == 4) & (verify_df["NumNeurons"] == 10)
    depth = verify_df[select].copy()
    # 4 x 10 inputs -> size times 10 x 10 -> 10 x 2 outputs
    depth["NumParameters"] = 40 + 100 * (depth["NumLayers"] - 1) + 20
    # depth.replace("TO", 3600, inplace=True)
    # depth.sort_values(by="NumParameters", inplace=True)
    # out_file = experiment_directory / "MiniACSIncomeNumLayers.csv"
    # print("Writing", out_file)
    # depth.to_csv(out_file, index=False)

    df = pd.concat([depth, width], axis=0)
    df.replace("TO", 3600, inplace=True)
    df.sort_values(by="NumParameters", inplace=True)
    out_file = experiment_directory / "MiniACSIncomeNumParameters.csv"
    print("Writing", out_file)
    df.to_csv(out_file, index=False)
