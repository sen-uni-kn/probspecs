# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import pytest
from pytest import approx

import torch

from probspecs import *


def test_construct_formula_1():
    f = ExternalFunction("f", ("x",))
    g = ExternalFunction("g", ("y", "z"))
    formula = (f + 5.0) * 3.5 >= g / 2.0
    print(formula)


def test_construct_formula_2():
    f = ExternalFunction("f", ("x",))
    g = ExternalFunction("g", ("y", "z"))
    formula = prob((f + 5.0) * 3.5 != g / 2.0) >= 0.75
    print(formula)


def test_construct_formula_3():
    net = ExternalFunction("net", ("x",))
    x = ExternalVariable("x")
    protected = x[28] >= 1.0
    hire = net[0] >= net[1]
    formula = prob(hire, condition=protected) / prob(hire, condition=~protected) >= 0.8
    print(formula)


def test_construct_formula_4():
    net = ExternalFunction("net", ("x",))
    x = ExternalVariable("x")
    protected = x[28] >= 1.0
    qualified = (x[12] >= 1.0) | (x[40] >= 1.0)
    hire = net[0] >= net[1]
    formula_1 = (
        prob(hire, condition=protected) / prob(hire, condition=~protected) >= 0.8
    )
    formula_2 = (
        prob(hire, condition=protected & qualified)
        / prob(hire, condition=~protected & qualified)
        >= 0.8
    )
    print(formula_1 | formula_2)


def test_eval_formula_1():
    f = ExternalFunction("f", ("x",))
    expression = prob(f >= 0)
    formula = expression >= 0.5

    def func(x_):
        return 2 * x_

    x = torch.tensor([-1, -2, 1, 2])

    val_expression = expression(f=func, x=x)
    assert val_expression == approx(0.5)
    assert formula(f=func, x=x)


if __name__ == "__main__":
    pytest.main()
