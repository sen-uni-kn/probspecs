# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Literal

import scipy.stats
import torch
from torch import nn

from probspecs import ExternalVariable, ExternalFunction, Probability
from probspecs import TensorInputSpace, TabularInputSpace
from probspecs import ToTensor
from probspecs.bounds.probability_bounds import probability_bounds

import pytest


@pytest.fixture
def first_scenario_1d():
    # 1d input space of a normally distributed random variable
    input_space = TensorInputSpace(
        lbs=torch.tensor([-10.0]),
        ubs=torch.tensor([10.0]),
    )
    distribution = ToTensor(scipy.stats.norm)

    # binary classifier
    # net produces the first class if the input is >= 0.0
    # and the second class otherwise
    net = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2))
    with torch.no_grad():
        net[0].weight.data = torch.tensor([[1.0], [-1.0]])
        net[0].bias.data = torch.zeros(2)
        net[2].weight.data = torch.eye(2)
        net[2].bias.data = torch.zeros(2)

    # binary classifier indicating if the input is >= 1.0
    net2 = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2))
    with torch.no_grad():
        net2[0].weight.data = torch.tensor([[1.0], [-1.0]])
        net2[0].bias.data = torch.tensor([-1.0, 1.0])
        net2[2].weight.data = torch.eye(2)
        net2[2].bias.data = torch.zeros(2)

    return net, net2, input_space, distribution


@pytest.mark.parametrize("split_heuristic", ["longest-edge"])
def test_probability_bounds_1(
    split_heuristic: Literal["longest-edge", "IBP"], first_scenario_1d
):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(251572694470273)
    net, _, input_space, distribution = first_scenario_1d

    net_func = ExternalFunction("net", ("x",))
    prob = Probability(net_func[:, 0] >= net_func[:, 1])

    bounds_gen = probability_bounds(
        prob,
        {"net": net},
        {"x": input_space},
        {"x": distribution},
        batch_size=4,
        split_heuristic=split_heuristic,
    )

    for i in range(25):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb <= 0.5
        assert ub >= 0.5


@pytest.mark.parametrize("split_heuristic", ["longest-edge"])
def test_probability_bounds_conditional_1(
    split_heuristic: Literal["longest-edge", "IBP"], first_scenario_1d
):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(249883610130386)
    net, _, input_space, distribution = first_scenario_1d

    x = ExternalVariable("x")
    net_func = ExternalFunction("net", ("x",))
    prob = Probability(net_func[:, 0] >= net_func[:, 1], condition=x >= 0.0)

    bounds_gen = probability_bounds(
        prob,
        {"net": net},
        {"x": input_space},
        {"x": distribution},
        batch_size=4,
        split_heuristic=split_heuristic,
    )

    for i in range(25):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb <= 1.0
        assert ub >= 1.0


@pytest.mark.parametrize("split_heuristic", ["longest-edge"])
def test_probability_bounds_conditional_2(
    split_heuristic: Literal["longest-edge", "IBP"], first_scenario_1d
):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(249883610130386)
    net, net2, input_space, distribution = first_scenario_1d

    net_func = ExternalFunction("net", ("x",))
    net2_func = ExternalFunction("net2", ("x",))
    prob = Probability(
        net_func[:, 0] >= net_func[:, 1], condition=net2_func[:, 0] >= net2_func[:, 1]
    )

    bounds_gen = probability_bounds(
        prob,
        {"net": net, "net2": net2},
        {"x": input_space},
        {"x": distribution},
        batch_size=64,
        split_heuristic=split_heuristic,
    )

    for i in range(10):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb <= 1.0
        assert ub >= 1.0
