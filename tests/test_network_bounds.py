# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Literal

import torch
from torch import nn

from probspecs.bounds.network_bounds import NetworkBounds

import pytest


@pytest.mark.parametrize("split_heuristic", ["longest-edge", "IBP"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.xfail(
                condition=not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_refine_bounds(split_heuristic: Literal["longest-edge", "IBP"], device: str):
    torch.manual_seed(883267399462969)
    net = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))

    in_lb = torch.zeros(10)
    in_ub = torch.ones(10)

    test_inputs = torch.rand((100, 10))
    test_outputs = net(test_inputs).to(device)

    compute_bounds = NetworkBounds(
        batch_size=256, split_heuristic=split_heuristic, device=device
    )
    bounds_gen = compute_bounds.bound(
        net,
        (in_lb, in_ub),
    )
    best_lb = -torch.inf
    best_ub = torch.inf
    for i in range(25):
        new_best_lb, new_best_ub = next(bounds_gen)

        assert torch.all(new_best_lb.ge(best_lb) | new_best_lb.isclose(new_best_lb))
        assert torch.all(new_best_ub.le(best_ub) | new_best_ub.isclose(new_best_ub))
        best_lb, best_ub = new_best_lb, new_best_ub

        assert torch.all(best_lb.le(test_outputs))
        assert torch.all(best_ub.ge(test_outputs))

        if i % 5 == 0 or i == 24:
            print(f"{i:3} lb: {best_lb}; ub: {best_ub}")
