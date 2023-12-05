# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch

from probspecs import (
    ExternalVariable,
    ExternalFunction,
    ExplicitFunction,
    Composition,
    compose,
)
from probspecs.utils.formula_utils import fuse_compositions

import pytest


x = ExternalVariable("x")
y = ExternalVariable("y")
f = ExternalFunction("f", ("z",))
g = ExternalFunction("g", ("z",))
h = ExternalFunction("h", ("z", "w"))

expression1 = 0.5 * compose(f, z=x + y) / y + 1.5
expression2 = 0.99 - compose(f, z=compose(g, z=x)) * y
expression3 = 2 / compose(h, z=expression2, w=compose(f, z=compose(f, z=x)))


@pytest.mark.parametrize("expression", [expression1, expression2, expression3])
def test_fuse_compositions(expression):
    torch.manual_seed(108654752447368)

    fused, subs = fuse_compositions(expression)
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
        expression(x=x, y=y, f=f, g=g, h=h), fused(x=x, y=y, f=f, g=g, h=h)
    )


relu = torch.nn.ReLU()
net = ExplicitFunction("net", ("z",), relu)
expression4 = compose(net, z=x * y)
expression5 = compose(g, z=expression4)


@pytest.mark.parametrize("expression", [expression4, expression5])
def test_fuse_compositions_2(expression):
    fused, _ = fuse_compositions(expression)
    print(fused)
    assert isinstance(fused, ExplicitFunction)
    assert isinstance(fused.func, torch.nn.Module)
