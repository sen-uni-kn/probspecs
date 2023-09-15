# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
from torch import nn

from probspecs.network_bounds import refine_bounds

import pytest


@pytest.mark.parametrize("split_heuristic", ["longest-edge", "ibp"])
def test_refine_bounds(split_heuristic):
    torch.manual_seed(883267399462969)
    net = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))

    in_lb = torch.zeros(10)
    in_ub = torch.ones(10)

    test_inputs = torch.rand((100, 10))
    test_outputs = net(test_inputs)

    bounds_gen = refine_bounds(
        net, (in_lb, in_ub), batch_size=256, split_heuristic=split_heuristic
    )
    for i in range(100):
        best_lb, best_ub = next(bounds_gen)
        assert torch.all(best_lb.le(test_outputs))
        assert torch.all(best_ub.ge(test_outputs))

        if i % 10 == 9:
            print(f"{i:3} lb: {best_lb}; ub: {best_ub}")
