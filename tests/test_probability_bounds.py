# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch

from probspecs import ExternalVariable, ExternalFunction, Probability, compose
from probspecs.bounds.probability_bounds import (
    SPLIT_HEURISTICS,
    ProbabilityBounds,
)

import pytest


@pytest.mark.parametrize("split_heuristic", SPLIT_HEURISTICS)
def test_probability_bounds_1(split_heuristic, verification_test_nets_1d):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(251572694470273)
    net, _, input_space, distribution = verification_test_nets_1d

    net_func = ExternalFunction("net", ("x",))
    prob = Probability(net_func[:, 0] >= net_func[:, 1])

    compute_bounds = ProbabilityBounds(batch_size=4, split_heuristic=split_heuristic)
    bounds_gen = compute_bounds.bound(
        prob, {"net": net}, {"x": input_space}, {"x": distribution}
    )

    for i in range(25):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb <= 0.5
        assert ub >= 0.5


@pytest.mark.parametrize("split_heuristic", SPLIT_HEURISTICS)
def test_probability_bounds_conditional_1(split_heuristic, verification_test_nets_1d):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(249883610130386)
    net, _, input_space, distribution = verification_test_nets_1d

    x = ExternalVariable("x")
    net_func = ExternalFunction("net", ("x",))
    prob = Probability(net_func[:, 0] >= net_func[:, 1], condition=x >= 0.0)

    compute_bounds = ProbabilityBounds()
    compute_bounds.configure(batch_size=4, split_heuristic=split_heuristic)
    bounds_gen = compute_bounds.bound(
        prob,
        {"net": net},
        {"x": input_space},
        {"x": distribution},
    )

    for i in range(25):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb <= 1.0
        assert ub >= 1.0


@pytest.mark.parametrize("split_heuristic", SPLIT_HEURISTICS)
def test_probability_bounds_conditional_2(split_heuristic, verification_test_nets_1d):
    """
    Test computing bounds on previously known probabilities
    """
    torch.manual_seed(249883610130386)
    net, _, input_space, distribution = verification_test_nets_1d

    x = ExternalVariable("x")
    net_func = ExternalFunction("net", ("x",))
    prob = Probability(
        net_func[:, 0] >= net_func[:, 1], condition=(x >= -1.0) & (x <= 1.0)
    )

    compute_bounds = ProbabilityBounds()
    compute_bounds.configure(batch_size=4, split_heuristic=split_heuristic)
    bounds_gen = compute_bounds.bound(
        prob,
        {"net": net},
        {"x": input_space},
        {"x": distribution},
    )

    for i in range(25):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb <= 0.5
        assert ub >= 0.5


@pytest.mark.parametrize("split_heuristic", ["longest-edge", "IBP"])
def test_probability_bounds_conditional_3(split_heuristic, verification_test_nets_1d):
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

    compute_bounds = ProbabilityBounds()
    compute_bounds.configure(batch_size=4, split_heuristic=split_heuristic)
    bounds_gen = compute_bounds.bound(
        prob,
        {"net": net, "net2": net2},
        {"x": input_space},
        {"x": distribution},
    )

    for i in range(25):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb <= 1.0
        assert ub >= 1.0


@pytest.mark.parametrize("split_heuristic", SPLIT_HEURISTICS)
def test_probability_bounds_compose_1(
    split_heuristic,
    verification_test_compose,
):
    input_space, distribution, generator, consumer = verification_test_compose

    g = ExternalFunction("g", ("x",))
    c = ExternalFunction("c", ("z",))
    cg = compose(c, z=g)
    prob = Probability(cg[:, 0] >= 0.31)

    compute_bounds = ProbabilityBounds(batch_size=64, split_heuristic=split_heuristic)
    bounds_gen = compute_bounds.bound(
        prob,
        {"g": generator, "c": consumer},
        {"x": input_space},
        {"x": distribution},
    )

    prev_lb = -torch.inf
    prev_ub = torch.inf
    for i in range(100):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb >= prev_lb - 1e-7  # floating-point issues
        assert ub <= prev_ub + 1e-7
        prev_lb, prev_ub = lb, ub


@pytest.mark.parametrize("split_heuristic", ["longest-edge", "IBP"])
@pytest.mark.xfail  # ConvTranspose and CROWN.
def test_probability_bounds_mnist_gen_1(
    split_heuristic,
    verification_test_mnist_conv_gen,
):
    gen_input_space, gen_distribution, generator = verification_test_mnist_conv_gen

    g = ExternalFunction("g", ("x",))
    prob = Probability(g[:, 0] >= 0.25)

    compute_bounds = ProbabilityBounds(
        batch_size=64, split_heuristic=split_heuristic, auto_lirpa_method="CROWN"
    )
    bounds_gen = compute_bounds.bound(
        prob,
        {"g": generator},
        {"x": gen_input_space},
        {"x": gen_distribution},
    )

    prev_lb = -torch.inf
    prev_ub = torch.inf
    for i in range(25):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb >= prev_lb - 1e-7  # floating-point issues
        assert ub <= prev_ub + 1e-7
        prev_lb, prev_ub = lb, ub


@pytest.mark.parametrize("split_heuristic", SPLIT_HEURISTICS)
@pytest.mark.parametrize("input_size", [3.0, 0.1], ids=("large", "small"))
def test_probability_bounds_mnist_1(
    split_heuristic,
    verification_test_mnist_fcnn_gen,
    small_conv_mnist_net,
    input_size,
):
    gen_input_space, gen_distribution, generator = verification_test_mnist_fcnn_gen
    classifier = small_conv_mnist_net

    x = ExternalVariable("x")
    g = ExternalFunction("g", ("x",))
    c = ExternalFunction("c", ("z",))
    cg = compose(c, z=g)
    class_two = (
        (cg[:, 2] >= cg[:, 0])
        & (cg[:, 2] >= cg[:, 1])
        & (cg[:, 2] >= cg[:, 3])
        & (cg[:, 2] >= cg[:, 4])
        & (cg[:, 2] >= cg[:, 5])
        & (cg[:, 2] >= cg[:, 6])
        & (cg[:, 2] >= cg[:, 7])
        & (cg[:, 2] >= cg[:, 8])
        & (cg[:, 2] >= cg[:, 9])
    )
    input_space = (
        (x[:, 0] >= -input_size)
        & (x[:, 0] <= input_size)
        & (x[:, 1] >= -input_size)
        & (x[:, 1] <= input_size)
        & (x[:, 2] >= -input_size)
        & (x[:, 2] <= input_size)
        & (x[:, 3] >= -input_size)
        & (x[:, 3] <= input_size)
    )
    # probability that classifier produces "2" for images generated by the generator
    prob = Probability(class_two, condition=input_space)

    compute_bounds = ProbabilityBounds(batch_size=4, split_heuristic=split_heuristic)
    bounds_gen = compute_bounds.bound(
        prob,
        {"g": generator, "c": classifier},
        {"x": gen_input_space},
        {"x": gen_distribution},
    )

    prev_lb = -torch.inf
    prev_ub = torch.inf
    for i in range(20):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb >= prev_lb - 1e-7  # floating-point issues
        assert ub <= prev_ub + 1e-7
        prev_lb, prev_ub = lb, ub


@pytest.mark.parametrize("split_heuristic", ["longest-edge", "IBP"])
def test_probability_bounds_mnist_2(
    split_heuristic,
    verification_test_mnist_conv_gen,
    small_conv_mnist_net,
):
    gen_input_space, gen_distribution, generator = verification_test_mnist_conv_gen
    classifier = small_conv_mnist_net

    x = ExternalVariable("x")
    g = ExternalFunction("g", ("x",))
    c = ExternalFunction("c", ("z",))
    cg = compose(c, z=g)
    class_two = (
        (cg[:, 2] >= cg[:, 0])
        & (cg[:, 2] >= cg[:, 1])
        & (cg[:, 2] >= cg[:, 3])
        & (cg[:, 2] >= cg[:, 4])
        & (cg[:, 2] >= cg[:, 5])
        & (cg[:, 2] >= cg[:, 6])
        & (cg[:, 2] >= cg[:, 7])
        & (cg[:, 2] >= cg[:, 8])
        & (cg[:, 2] >= cg[:, 9])
    )
    input_space = (
        (x[:, 0, 0, 0] >= -3.0)
        & (x[:, 0, 0, 0] <= 3.0)
        & (x[:, 1, 0, 0] >= -3.0)
        & (x[:, 1, 0, 0] <= 3.0)
        & (x[:, 2, 0, 0] >= -3.0)
        & (x[:, 2, 0, 0] <= 3.0)
        & (x[:, 3, 0, 0] >= -3.0)
        & (x[:, 3, 0, 0] <= 3.0)
    )
    # probability that classifier produces "2" for images generated by the generator
    prob = Probability(class_two, condition=input_space)

    compute_bounds = ProbabilityBounds(
        batch_size=16,
        split_heuristic=split_heuristic,
        auto_lirpa_method="ibp",  # "CROWN" leads to error
    )
    bounds_gen = compute_bounds.bound(
        prob,
        {"g": generator, "c": classifier},
        {"x": gen_input_space},
        {"x": gen_distribution},
    )

    prev_lb = -torch.inf
    prev_ub = torch.inf
    for i in range(25):
        lb, ub = next(bounds_gen)
        print(f"lb={lb:.4f}, ub={ub:.4f}")
        assert lb >= prev_lb - 1e-7  # floating-point issues
        assert ub <= prev_ub + 1e-7
        prev_lb, prev_ub = lb, ub


@pytest.mark.xfail(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_probability_bounds_on_cuda(verification_test_nets_1d):
    """
    Test computing probability bounds on CUDA.
    """
    torch.manual_seed(50215201576157)
    net, _, input_space, distribution = verification_test_nets_1d

    net_func = ExternalFunction("net", ("x",))
    prob = Probability(net_func[:, 0] >= net_func[:, 1])

    compute_bounds = ProbabilityBounds(batch_size=16, device="cuda")
    bounds_gen = compute_bounds.bound(
        prob,
        {"net": net},
        {"x": input_space},
        {"x": distribution},
    )

    for i in range(5):
        next(bounds_gen)


if __name__ == "__main__":
    pytest.main()
