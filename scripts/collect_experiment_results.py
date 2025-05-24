#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import argparse
import os

import pandas as pd
import re
from glob import glob
from pathlib import Path


def read_last_lines(file_path, num_lines: int):
    """
    Read the last n lines of a file.
    """
    # Code from https://stackoverflow.com/a/73195814/10550998
    # by Jazz Weisman
    newline_counter = 0
    with open(file_path, "rb") as file:
        try:
            file.seek(-2, os.SEEK_END)
            while newline_counter < num_lines:
                file.seek(-2, os.SEEK_CUR)
                if file.read(1) == b"\n":
                    newline_counter += 1
        except OSError:
            file.seek(0)
        return [line.decode().replace("\n", "") for line in file.readlines()]


def list_log_files(dir_):
    log_files = glob("*.log", root_dir=dir_)
    log_files += glob("*.txt", root_dir=dir_)
    return log_files


def get_instance_name(dir_, file_name):
    for parent_dir in reversed(dir_.parts[-3:]):
        match parent_dir:
            case "fairsquare":
                pop_model, _, network, has_qual = file_name.split("_", maxsplit=3)
                has_qual, _ = has_qual.split(".")  # drop file extension
                return {
                    "Population Model": pop_model,
                    "Network": network,
                    "Qualified": has_qual == "qual",
                }
            case "mini_acs_income":
                num_vars, *remainder = file_name.split("_")
                info = {"Input Variables": num_vars}
                if len(remainder) == 1:
                    info["Num Neurons"] = 10
                    info["Num Layers"] = 1
                else:
                    _, size, kind = remainder
                    if kind.startswith("neurons"):
                        info["Num Neurons"] = int(size)
                        info["Num Layers"] = 1
                    elif kind.startswith("layers"):
                        info["Num Neurons"] = 10
                        info["Num Layers"] = int(size)
                    else:
                        raise ValueError(f"Unknown network: {file_name}.")
                return info
            case "acasxu":
                if "property" in file_name:
                    # example: property2_2_1.log
                    # example: property100_netABC.log
                    underscore_i = file_name.index("_")
                    prop = file_name[:underscore_i]
                    network, _ = file_name[underscore_i + 1 :].split(".")
                    return {"Network": network, "Property": prop}
                else:  # robustness experiment
                    # example: net1_1_0_to_0_0.log
                    # example: netABC_3_to_1_1999.log
                    network, source_label, _, target_label, end = file_name.rsplit(
                        "_", maxsplit=4
                    )
                    input_i, _ = end.split(".")
                    return {
                        "Network": network,
                        "Source Label": int(source_label),
                        "Target Label": int(target_label),
                        "Input": int(input_i),
                    }
    raise ValueError(f"Unknown experiment: {dir_}.")


runtime_re = re.compile(r"Runtime:? *(?P<seconds>\d+\.\d*)")


def collect_verify(dir_):
    data = []
    log_files = list_log_files(dir_)
    for log_file in log_files:
        entry = get_instance_name(dir_, log_file)
        last_lines = read_last_lines(dir_ / log_file, num_lines=3)
        if "VerifyStatus" not in last_lines[0] or "Runtime" not in last_lines[-1]:
            entry |= {
                "Fair": "Unknown",
                "Runtime": "TO",
            }
        else:
            is_fair = "SATISFIED" in last_lines[0]
            runtime = runtime_re.search(last_lines[-1]).group("seconds")
            entry |= {
                "Fair": is_fair,
                "Runtime": float(runtime),
            }
        data.append(entry)
    data_file = dir_ / "results.csv"
    print(f"Saving results from {dir_} to {data_file}.")
    data = pd.DataFrame(data)
    data.to_csv(data_file, index=False, float_format="%.8f")


bounds_re = re.compile(r"(?P<lb>-?\d+\.\d*) *<=.*<= *(?P<ub>-?\d+\.\d*)")


def collect_bound(dir_):
    data = []
    log_files = list_log_files(dir_)
    for log_file in log_files:
        entry = get_instance_name(dir_, log_file)
        last_lines = read_last_lines(dir_ / log_file, num_lines=3)
        bounds_match = bounds_re.search(last_lines[0]) if len(last_lines) > 0 else None
        runtime_match = (
            runtime_re.search(last_lines[2]) if len(last_lines) > 1 else None
        )
        if len(last_lines) < 3 or bounds_match is None or runtime_match is None:
            print(f"Skipping incomplete log file: {dir_ / log_file}.")
        else:
            entry["Lower Bound"] = float(bounds_match.group("lb"))
            entry["Upper Bound"] = float(bounds_match.group("ub"))
            entry["Runtime"] = float(runtime_match.group("seconds"))
            entry["Abort Reason"] = last_lines[1].replace(".", "").strip()
            data.append(entry)
    data_file = dir_ / "results.csv"
    print(f"Saving results from {dir_} to {data_file}.")
    data = pd.DataFrame(data)
    data.to_csv(data_file, index=False, float_format="%.8f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect Experiment Results")
    parser.add_argument(
        "directory",
        type=str,
        help="The directory containing the experiment results. "
        "For example experiments/output/SOME_TIMESTAMP/",
    )
    args = parser.parse_args()
    experiment_directory = Path(args.directory)
    print(f"Scanning {experiment_directory} for experiment results.")
    subdirs = [d for d in experiment_directory.iterdir() if d.is_dir()]
    for subdir in subdirs:
        match subdir.name:
            case "fairsquare":
                collect_verify(subdir)
            case "mini_acs_income":
                if (subdir / "verify").exists():
                    collect_verify(subdir / "verify")
                if (subdir / "enumerate").exists():
                    collect_verify(subdir / "enumerate")
            case "acasxu":
                if (subdir / "robustness").exists():
                    collect_bound(subdir / "robustness")
                safety_dir = subdir / "safety"
                if safety_dir.exists():
                    for timeout_dir in safety_dir.iterdir():
                        if timeout_dir.is_dir():
                            collect_bound(timeout_dir)
            case _:
                print(f"Skipping unknown directory: {subdir.name}")
