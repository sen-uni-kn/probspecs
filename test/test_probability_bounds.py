# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Literal

import torch

from probspecs import ExternalVariable, ExternalFunction, Probability
from probspecs.bounds.probability_bounds import probability_bounds

import pytest


@pytest.mark.parametrize("split_heuristic", ["longest-edge"])
def test_probability_bounds_1(
    split_heuristic: Literal["longest-edge", "IBP"], verification_test_nets_1d
):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(251572694470273)
    net, _, input_space, distribution = verification_test_nets_1d

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
    split_heuristic: Literal["longest-edge", "IBP"], verification_test_nets_1d
):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(249883610130386)
    net, _, input_space, distribution = verification_test_nets_1d

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
    split_heuristic: Literal["longest-edge", "IBP"], verification_test_nets_1d
):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(249883610130386)
    net, net2, input_space, distribution = verification_test_nets_1d

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
