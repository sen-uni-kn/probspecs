#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import argparse
from collections import defaultdict
from math import floor
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import random_split

from miniacsincome import MiniACSIncome


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate MiniACSIncome Networks")
    parser.add_argument("networks", nargs="*", type=str)
    args = parser.parse_args()

    networks = defaultdict(list)
    for path in args.networks:
        path = Path(path)
        # drop .pyt extension and MiniACSIncome prefix
        _, name = path.name[:-4].split("-")
        num_variables, *remainder = name.split("_")
        num_layers = 1
        num_neurons = 10
        if len(remainder) > 1:
            kind, size = remainder
            if kind == "size":
                num_neurons = int(size)
            elif kind == "depth":
                num_layers = int(size)
            else:
                raise ValueError(f"Unknown network: {path}.")

        net = torch.load(path)
        networks[int(num_variables)].append((num_layers, num_neurons, net))

    stats_data = []

    for num_variables, nets in networks.items():
        print()
        print("=" * 100)
        print(f"MiniACSIncome-{num_variables}")
        print("=" * 100)
        print()
        print()

        dataset = MiniACSIncome(
            root=".datasets", num_variables=num_variables, normalize=True
        )
        rng = torch.Generator().manual_seed(163050276766629)
        train_size = floor(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        _, dataset = random_split(dataset, (train_size, test_size), generator=rng)

        data = dataset.dataset.data[dataset.indices]
        targets = dataset.dataset.targets[dataset.indices]

        @torch.no_grad()
        def stats(net_, data=data, targets=targets):
            predictions = torch.argmax(net_(data), dim=1)
            tp_rate = ((predictions == 1) & (targets == 1)).float().mean()
            fp_rate = ((predictions == 1) & (targets == 0)).float().mean()
            tn_rate = ((predictions == 0) & (targets == 0)).float().mean()
            fn_rate = ((predictions == 0) & (targets == 1)).float().mean()
            accuracy = tp_rate + tn_rate
            precision = tp_rate / (tp_rate + fp_rate)
            recall = tp_rate / (tp_rate + fn_rate)
            fscore = 2 * (precision * recall) / (precision + recall)
            return accuracy, precision, recall, fscore

        encoding_layout = dataset.dataset.input_space.encoding_layout
        female_i = encoding_layout["SEX"]["Female"]
        male_i = encoding_layout["SEX"]["Male"]
        female = data[:, female_i] >= 1.0
        male = data[:, male_i] >= 1.0
        female_data, female_targets = data[female], targets[female]
        male_data, male_targets = data[male], targets[male]

        @torch.no_grad()
        def stats_by_sex(net_):
            female_stats = stats(net_, female_data, female_targets)
            male_stats = stats(net_, male_data, male_targets)
            return female_stats, male_stats

        nets.sort(key=lambda entry: entry[:2])

        for num_layers, num_neurons, net in nets:
            accuracy, precision, recall, fscore = stats(net)
            f_stats, m_stats = stats_by_sex(net)
            f_accuracy, f_precision, f_recall, f_fscore = f_stats
            m_accuracy, m_precision, m_recall, m_fscore = m_stats
            print(
                f"Network with Input Size: {num_variables}, Num Layers: {num_layers}, "
                f"Num Neurons: {num_neurons}"
            )
            print("-" * 100)
            print(f"        | Accuracy  | Precision | Recall    | F-Score   ")
            print(f"--------|-----------|-----------|-----------|-----------")
            print(
                f"Overall | {accuracy:9.2%} | {precision:9.2%} "
                f"| {recall:9.2%} | {fscore:9.2f}"
            )
            print(
                f"Female  | {f_accuracy:9.2%} | {f_precision:9.2%} "
                f"| {f_recall:9.2%} | {f_fscore:9.2f}"
            )
            print(
                f"Male    | {m_accuracy:9.2%} | {m_precision:9.2%} "
                f"| {m_recall:9.2%} | {m_fscore:9.2f}"
            )
            print()

            stats_data.append(
                {
                    "Input Size": num_variables,
                    "Num Layers": num_layers,
                    "Num Neurons": num_neurons,
                    "Overall Accuracy": accuracy.item(),
                    "Overall Precision": precision.item(),
                    "Overall Recall": recall.item(),
                    "Overall F-Score": fscore.item(),
                    "Female Accuracy": f_accuracy.item(),
                    "Female Precision": f_precision.item(),
                    "Female Recall": f_recall.item(),
                    "Female F-Score": f_fscore.item(),
                    "Male Accuracy": m_accuracy.item(),
                    "Male Precision": m_precision.item(),
                    "Male Recall": m_recall.item(),
                    "Male F-Score": m_fscore.item(),
                }
            )

    stats_df = pd.DataFrame(stats_data)

    def format_float(val):
        return rf"{val * 100:3.0f}\%"

    print()
    print()
    print(
        stats_df.to_latex(
            formatters={
                "Overall Accuracy": format_float,
                "Overall Precision": format_float,
                "Overall Recall": format_float,
                "Female Accuracy": format_float,
                "Female Precision": format_float,
                "Female Recall": format_float,
                "Male Accuracy": format_float,
                "Male Precision": format_float,
                "Male Recall": format_float,
            }
        )
    )
