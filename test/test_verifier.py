# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from functools import reduce

import torch

from probspecs import (
    ExternalVariable,
    ExternalFunction,
    Composition,
    Probability,
    compose,
    verify,
)
from probspecs.verifier import fuse_compositions, VerificationStatus

import pytest


def test_verifier_1(verification_test_nets_1d):
    torch.manual_seed(6982588805995)
    # net1 is a binary classifier for x >= 0
    # net2 is a binary classifier for x >= 1
    # input x is from [-10, 10] with a standard normal distribution
    net1, net2, input_space, distribution = verification_test_nets_1d

    net1_func = ExternalFunction("net1", ("x",))
    net2_func = ExternalFunction("net2", ("x",))
    # probability x >= 0
    prob1 = Probability(net1_func[:, 0] >= net1_func[:, 1])
    # probability x >= 1
    prob2 = Probability(net2_func[:, 0] >= net2_func[:, 1])

    formula = prob1 >= prob2  # True
    verification_status, bounds = verify(
        formula, {"net1": net1, "net2": net2}, {"x": input_space}, {"x": distribution}
    )
    assert verification_status is VerificationStatus.SATISFIED
    assert bounds[prob1][0] >= bounds[prob2][1]  # lb >= ub
    print(bounds)


def test_verifier_2(verification_test_nets_1d):
    torch.manual_seed(6982588305995)
    net1, net2, input_space, distribution = verification_test_nets_1d

    net1_func = ExternalFunction("net1", ("x",))
    net2_func = ExternalFunction("net2", ("x",))
    # P(x >= 0) = 0.5
    prob1 = Probability(net1_func[:, 0] >= net1_func[:, 1])
    # P(x >= 1) = 0.1587 (approx)
    prob2 = Probability(net2_func[:, 0] >= net2_func[:, 1])

    formula = prob2 / prob1 >= 0.35  # False
    verification_status, bounds = verify(
        formula, {"net1": net1, "net2": net2}, {"x": input_space}, {"x": distribution}
    )
    assert verification_status is VerificationStatus.VIOLATED
    print(bounds)


def test_verifier_mnist_1(verification_test_mnist_gen, small_conv_mnist_net):
    gen_input_space, gen_distribution, generator = verification_test_mnist_gen
    classifier = small_conv_mnist_net

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
    # probability that classifier produces "2" for images generated by the generator
    formula = Probability(class_two) >= 0.5

    verification_status, bounds = verify(
        formula,
        {"g": generator, "c": classifier},
        {"x": gen_input_space},
        {"x": gen_distribution},
    )
    assert verification_status in (
        VerificationStatus.SATISFIED,
        VerificationStatus.VIOLATED,
    )
    print(bounds)


x = ExternalVariable("x")
y = ExternalVariable("y")
f = ExternalFunction("f", ("z",))
g = ExternalFunction("g", ("z",))
h = ExternalFunction("h", ("z", "w"))

expression1 = 0.5 * compose(f, z=x + y) / y + 1.5
expression2 = 0.99 - compose(f, z=compose(g, z=x)) * y
expression3 = 2 / compose(h, z=expression2, w=compose(f, z=compose(f, z=x)))


@pytest.mark.parametrize("formula", [expression1, expression2, expression3])
def test_fuse_compositions(formula):
    torch.manual_seed(108654752447368)

    fused, subs = fuse_compositions(formula)
    print(fused)
    print(subs)
    assert len(fused.collect(lambda sub: isinstance(sub, Composition))) == 0

    x = 100 * torch.rand((100, 10)) - 50
    y = 100 * torch.rand((100, 10)) - 50

    def f(z):
        return torch.square(z)

    def g(z):
        return torch.sin(z)

    def h(z, w):
        return torch.sqrt(torch.abs(z)) / w

    assert torch.allclose(
        formula(x=x, y=y, f=f, g=g, h=h), fused(x=x, y=y, f=f, g=g, h=h)
    )
