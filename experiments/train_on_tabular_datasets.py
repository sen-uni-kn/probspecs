#  Copyright (c) 2023-2024. David Boetius
#  Licensed under the MIT License
from argparse import ArgumentParser
from math import floor
from pathlib import Path
import random
from typing import Callable

from fairnessdatasets import Adult, SouthGerman
import numpy as np
import optuna
from yaml import safe_dump, safe_load
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from miniacsincome import MiniACSIncome


if __name__ == "__main__":
    torch.manual_seed(962912072501243)
    rng = torch.Generator().manual_seed(163050276766629)
    np.random.seed(2465716661)
    random.seed(602560181477205)

    parser = ArgumentParser(
        "Train a neural network using Optuna hyperparameter search."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["Adult", "SouthGerman"] + [f"MiniACSIncome-{i}" for i in range(1, 11)],
        help="The dataset for which to train the network",
    )
    parser.add_argument(
        "--out", required=True, type=str, help="Where to save the trained network."
    )
    tuning_group = parser.add_argument_group("Hyperparameter Tuning")
    tuning_group.add_argument(
        "--trials",
        default=None,
        type=lambda x: None if x == "None" else int(x),
        help="How many trials of Optuna hyperparameter search to perform.",
    )
    tuning_group.add_argument(
        "--timeout",
        default=3600,
        type=int,
        help="How long to perform Optuna hyperparameter search in seconds.",
    )
    tuning_group.add_argument(
        "--jobs",
        default=1,
        type=int,
        help="The number of parallel jobs to use for hyperparameter search.",
    )
    tuning_group.add_argument(
        "--architecture",
        type=int,
        nargs="+",
        default=None,
        help="The network architecture to use. If None, the architecture is optimised "
        "during hyperparameter search. "
        "For example '--architecture 100 100 100' describes a neural network with three "
        "hidden layers that each contain 100 neurons. "
        "The network also contains an additional linear output layer of suitable size.",
    )
    tuning_group.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="The number of epochs to perform during training. "
        "If None, the number of epochs is optimised during hyperparameter search.",
    )
    tuning_group.add_argument(
        "--lr",
        type=float,
        default=None,
        help="The learning rate to use. If None, the learning rate is optimised "
        "during hyperparameter search.",
    )
    tuning_group.add_argument(
        "--beta1",
        type=float,
        default=None,
        help="The first beta coefficient of the Adam algorithm. See "
        "https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#adam "
        "for more information. "
        "If None, this coefficient is optimised during hyperparameter search.",
    )
    tuning_group.add_argument(
        "--beta2",
        type=float,
        default=None,
        help="The second beta coefficient of the Adam algorithm. See "
        "https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#adam "
        "for more information. "
        "If None, this coefficient is optimised during hyperparameter search.",
    )
    tuning_group.add_argument(
        "--lr-gamma",
        type=float,
        default=None,
        help="The learning rate decay of the learning rate scheduler. "
        "If None, the decay rate is optimised during hyperparameter search.",
    )
    tuning_group.add_argument(
        "--lr-milestones",
        type=int,
        nargs=2,
        default=None,
        help="The learning rate schedule milestone. The learning rate is decayed "
        "every time one of the iterations in this list is reached"
        "If None, the milestones are optimised during hyperparameter search.",
    )
    tuning_group.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="The weight decay parameter of the Adam algorithm. See "
        "https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#adam "
        "for more information. "
        "If None, this value is optimised during hyperparameter search.",
    )
    tuning_save_reuse = tuning_group.add_mutually_exclusive_group()
    tuning_save_reuse.add_argument(
        "--save-hyperparameters",
        default=None,
        help="Where to save the result of hyperparameter tuning. "
        "By default, the results are not saved.",
    )
    tuning_save_reuse.add_argument(
        "--use-hyperparameters",
        default=None,
        dest="hyperparam_file",
        help="Skip parameter tuning and use the parameters from a previous run "
        "that were saved in the given file. ",
    )
    args = parser.parse_args()

    match args.dataset.split("-"):
        case ["Adult"]:
            train_set = Adult(root=".datasets", train=True, download=True)
            test_set = Adult(root=".datasets", train=False, download=True)
            num_classes = 2
            input_size = len(test_set.columns)
        case ["SouthGerman"]:
            dataset = SouthGerman(root=".datasets", download=True)
            # create a 70%/30% split
            train_size = floor(0.7 * len(dataset))
            test_size = len(dataset) - train_size
            train_set, test_set = random_split(
                dataset, (train_size, test_size), generator=rng
            )
            num_classes = 2
            input_size = len(dataset.columns)
        case ["MiniACSIncome", size]:
            dataset = MiniACSIncome(
                root=".datasets", num_variables=int(size), download=True, normalize=True
            )
            # create a 70%/30% split
            train_size = floor(0.7 * len(dataset))
            test_size = len(dataset) - train_size
            train_set, test_set = random_split(
                dataset, (train_size, test_size), generator=rng
            )
            num_classes = 2
            input_size = len(dataset.columns)
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}.")

    train_size = floor(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, (train_size, val_size), generator=rng)

    def choose_param(name: str, choose: Callable):
        """Evaluates and returns choose, if the argument `name` is None."""
        arg_value = getattr(args, name)
        return choose() if arg_value is None else arg_value

    def build_network(layer_sizes: tuple[int, ...]) -> nn.Module:
        layers = []
        prev_layer_size = input_size
        for layer_size in layer_sizes:
            lin_layer = nn.Linear(prev_layer_size, layer_size)
            layers.append(lin_layer)
            layers.append(nn.ReLU())
            prev_layer_size = layer_size
        layers.append(nn.Linear(prev_layer_size, num_classes))
        return nn.Sequential(*layers)

    def new_network(trial: optuna.Trial) -> nn.Module:
        if args.architecture is not None:
            return build_network(args.architecture)
        else:
            num_layers = trial.suggest_int("num_layers", 1, 4)
            layer_sizes = tuple(
                trial.suggest_int(f"layer_size_{i}", 2, 100) for i in range(num_layers)
            )
            return build_network(layer_sizes)

    def run_training(
        trial: optuna.Trial, log=False, check_prune=True, save_network=False
    ):
        random_seed = trial.suggest_categorical(
            "random_seed",
            (
                611998035180481,
                885962867484526,
                431409684549184,
                84095786053604,
                231661490005539,
            ),
        )
        torch.manual_seed(random_seed)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
        epoch_len = len(train_loader)
        num_epochs = choose_param(
            "num_epochs", lambda: trial.suggest_int("num_epochs", 5, 20)
        )

        network = new_network(trial)
        lr = choose_param("lr", lambda: trial.suggest_float("lr", 1e-6, 0.1))
        beta1 = choose_param("beta1", lambda: trial.suggest_float("beta1", 0.8, 1.0))
        beta2 = choose_param("beta2", lambda: trial.suggest_float("beta2", 0.9, 1.0))
        weight_decay = choose_param(
            "weight_decay", lambda: trial.suggest_float("weight_decay", 0.0, 1.0)
        )
        optimizer = torch.optim.Adam(
            network.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        lr_gamma = choose_param(
            "lr_gamma", lambda: trial.suggest_float("lr_gamma", 0.05, 0.99)
        )
        if args.lr_milestones is None:
            lr_milestone_1 = trial.suggest_int(
                "lr_milestone_1", 1, num_epochs * epoch_len
            )
            lr_milestone_2 = trial.suggest_int(
                "lr_milestone_2", 1, num_epochs * epoch_len
            )
            lr_milestones_ = (lr_milestone_1, lr_milestone_1 + lr_milestone_2)
        else:
            lr_milestones_ = args.lr_milestones

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones_, gamma=lr_gamma
        )
        loss_function = nn.CrossEntropyLoss()

        full_train_loader = DataLoader(train_set, batch_size=len(train_set))
        full_val_loader = DataLoader(val_set, batch_size=len(val_set))
        full_test_loader = DataLoader(test_set, batch_size=len(test_set))

        @torch.no_grad()
        def full_loss(data_loader):
            loss = 0.0
            for inputs, targets in iter(data_loader):
                loss += loss_function(network(inputs), targets)
            return loss / len(data_loader)

        @torch.no_grad()
        def accuracy(data_loader):
            acc = 0.0
            for inputs, targets in iter(data_loader):
                predictions = torch.argmax(network(inputs), dim=1)
                acc += (targets == predictions).float().mean()
            return acc / len(data_loader)

        @torch.no_grad()
        def confusion_matrix(data_loader):
            tp_rate = fp_rate = tn_rate = fn_rate = 0.0
            for inputs, targets in iter(data_loader):
                predictions = torch.argmax(network(inputs), dim=1)
                tp_rate += ((predictions == 1) & (targets == 1)).float().mean()
                fp_rate += ((predictions == 1) & (targets == 0)).float().mean()
                tn_rate += ((predictions == 0) & (targets == 0)).float().mean()
                fn_rate += ((predictions == 0) & (targets == 1)).float().mean()
            tp_rate = tp_rate / len(data_loader)
            fp_rate = fp_rate / len(data_loader)
            tn_rate = tn_rate / len(data_loader)
            fn_rate = fn_rate / len(data_loader)
            precision = tp_rate / (tp_rate + fp_rate)
            recall = tp_rate / (tp_rate + fn_rate)
            return tp_rate, fp_rate, tn_rate, fn_rate, precision, recall

        log_frequency = max(1, epoch_len // 10)
        for epoch in range(num_epochs):
            for i, (inputs, targets) in enumerate(iter(train_loader)):
                optimizer.zero_grad()
                loss = loss_function(network(inputs), targets)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                log_now = log and i % log_frequency == 0
                log_now = log_now or (
                    i == len(train_loader) - 1 and epoch == num_epochs - 1
                )
                if log_now:
                    full_train_loss = full_loss(full_train_loader)
                    full_val_loss = full_loss(full_val_loader)
                    train_acc = accuracy(full_train_loader)
                    val_acc = accuracy(full_val_loader)
                    test_acc = accuracy(full_test_loader)
                    log_string = (
                        f"[Epoch {epoch + 1}/{num_epochs} | {i / epoch_len * 100:2.0f}%] "
                        f"batch loss: {loss:3.4f}, "
                        f"loss (train/val): {full_train_loss:3.4f}/{full_val_loss:3.4f}, "
                        f"accuracy (train/val/test): "
                        f"{train_acc * 100:4.2f}%/{val_acc * 100:4.2f}%/{test_acc * 100:4.2f}%"
                    )
                    if num_classes == 2:
                        _, _, _, _, train_precision, train_recall = confusion_matrix(
                            full_train_loader
                        )
                        _, _, _, _, val_precision, val_recall = confusion_matrix(
                            full_val_loader
                        )
                        _, _, _, _, test_precision, test_recall = confusion_matrix(
                            full_test_loader
                        )
                        log_string += (
                            f", precision (train/val/test): "
                            f"{train_precision * 100:4.2f}%/{val_precision * 100:4.2f}%/{test_precision * 100:4.2f}%, "
                            f"recall (train/val/test): "
                            f"{train_recall * 100:4.2f}%/{val_recall * 100:4.2f}%/{test_recall * 100:4.2f}%"
                        )
                    print(log_string)

            if check_prune:
                full_val_loss = full_loss(full_val_loader)
                trial.report(full_val_loss, (epoch + 1) * epoch_len)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if save_network:
            export_path = Path(args.out)
            print(f"Saving network in {args.out}")
            torch.save(network, export_path)

        return full_loss(full_val_loader)

    if not args.hyperparam_file:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            run_training, n_trials=args.trials, timeout=args.timeout, n_jobs=args.jobs
        )
        best_params = study.best_params

        if args.architecture is not None:
            best_params["num_layers"] = len(args.architecture)
            for i, layer_size in enumerate(args.architecture):
                best_params[f"layer_size_{i}"] = layer_size
        if args.num_epochs is not None:
            best_params["num_epochs"] = args.num_epochs
        if args.lr is not None:
            best_params["lr"] = args.lr
        if args.beta1 is not None:
            best_params["beta1"] = args.beta1
        if args.beta2 is not None:
            best_params["beta2"] = args.beta2
        if args.lr_gamma is not None:
            best_params["lr_gamma"] = args.lr_gamma
        if args.lr_milestones is not None:
            assert len(args.lr_milestones) == 2
            best_params["lr_milestone_1"] = args.lr_milestones[0]
            best_params["lr_milestone_2"] = args.lr_milestones[1]
        if args.weight_decay is not None:
            best_params["weight_decay"] = args.weight_decay

        if args.save_hyperparameters is not None:
            with open(args.save_hyperparameters, "wt") as param_file:
                print(f"Saving hyperparameters in {args.save_hyperparameters}.")
                safe_dump(best_params, param_file)
    else:
        with open(args.hyperparam_file, "rt") as param_file:
            print(f"Loading hyperparameters from {args.hyperparam_file}.")
            best_params = safe_load(param_file)

    print(f"Re-running best trial: {best_params}")
    best_trial = optuna.trial.FixedTrial(best_params)
    run_training(best_trial, log=True, check_prune=False, save_network=True)
