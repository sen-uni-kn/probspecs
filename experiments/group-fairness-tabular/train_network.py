# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from argparse import ArgumentParser
from math import floor
from pathlib import Path

from adult import Adult
import numpy as np
import optuna
from yaml import safe_dump, safe_load
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader


if __name__ == "__main__":
    torch.manual_seed(962912072501243)
    rng = torch.Generator().manual_seed(163050276766629)
    np.random.seed(2465716661)

    parser = ArgumentParser(
        "Train a neural network using Optuna hyperparameter search."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["Adult"],
        help="The dataset for which to train the network",
    )
    parser.add_argument(
        "--out", required=True, type=str, help="Where to save the trained network."
    )
    tuning_group = parser.add_argument_group("Hyperparameter Tuning")
    tuning_group.add_argument(
        "--trials",
        default=1000,
        type=int,
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
    tuning_save_reuse = parser.add_mutually_exclusive_group()
    tuning_save_reuse.add_argument(
        "--save-to",
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

    match args.dataset:
        case "Adult":
            train_set = Adult(root=".datasets", train=True, download=True)
            test_set = Adult(root=".datasets", train=False, download=True)
            num_classes = 2
            class_weights = (0.75, 0.25)
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}.")

    train_size = floor(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, (train_size, val_size), generator=rng)

    def new_network(trial: optuna.Trial) -> nn.Module:
        num_layers = trial.suggest_int("num_layers", 1, 4)
        layers = []
        prev_layer_size = len(test_set.columns)  # input size
        for i in range(num_layers):
            layer_size = trial.suggest_int(f"layer_size_{i}", 20, 1000)
            lin_layer = nn.Linear(prev_layer_size, layer_size)
            layers.append(lin_layer)
            layers.append(nn.ReLU())
            prev_layer_size = layer_size
        layers.append(nn.Linear(prev_layer_size, num_classes))
        return nn.Sequential(*layers)

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
        num_epochs = trial.suggest_int("num_epochs", 5, 20)

        network = new_network(trial)
        lr = trial.suggest_float("lr", 1e-6, 1.0, log=True)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        lr_gamma = trial.suggest_float("lr_gamma", 0.1, 0.9)
        lr_milestone_1 = trial.suggest_int("lr_milestone_1", 1, num_epochs * epoch_len)
        lr_milestone_2 = trial.suggest_int("lr_milestone_2", 1, num_epochs * epoch_len)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=(lr_milestone_1, lr_milestone_1 + lr_milestone_2),
            gamma=lr_gamma,
        )
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

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

        log_frequency = epoch_len // 10
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
                    print(
                        f"[Epoch {epoch+1}/{num_epochs} | {i / epoch_len * 100:2.0f}%] "
                        f"batch loss: {loss:3.4f}, "
                        f"loss (train/val): {full_train_loss:3.4f}/{full_val_loss:3.4f}, "
                        f"accuracy (train/val/test): "
                        f"{train_acc*100:4.2f}%/{val_acc*100:4.2f}%/{test_acc*100:4.2f}%"
                    )

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

        if args.save_to is not None:
            with open(args.save_to, "wt") as param_file:
                print(f"Saving hyperparameters in {args.save_to}.")
                safe_dump(best_params, param_file)
    else:
        with open(args.hyperparam_file, "rt") as param_file:
            print(f"Loading hyperparameters from {args.hyperparam_file}.")
            best_params = safe_load(param_file)

    print(f"Re-running best trial: {best_params}")
    best_trial = optuna.trial.FixedTrial(best_params)
    run_training(best_trial, log=True, check_prune=False, save_network=True)
