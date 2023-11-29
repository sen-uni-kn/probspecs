# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch

from probspecs import ExternalVariable, ExternalFunction, Composition, compose
from probspecs.verifier import fuse_compositions

import pytest


x = ExternalVariable("x")
y = ExternalVariable("y")
f = ExternalFunction("f", ("z",))
g = ExternalFunction("g", ("z",))
h = ExternalFunction("h", ("z", "w"))

formula1 = 0.5 * compose(f, z=x + y) / y + 1.5
formula2 = 0.99 - compose(f, z=compose(g, z=x)) * y
formula3 = 2 / compose(h, z=formula2, w=compose(f, z=compose(f, z=x)))


@pytest.mark.parametrize("formula", [formula1, formula2, formula3])
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
