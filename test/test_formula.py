# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import pytest
from pytest import approx

import torch

from probspecs import *
from probspecs import TrinaryLogic as TL
from probspecs.formula import max_expr, min_expr


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


def test_construct_formula_5():
    net = ExternalFunction("net", ("x",))
    x = ExternalVariable("x")
    formula = -net
    print(formula)


def test_construct_formula_6():
    net = ExternalFunction("net", ("x",))
    x = ExternalVariable("x")
    y = ExternalVariable("y")
    z = ExternalVariable("z")
    formula = max_expr(min_expr(x, y), z, net)
    print(formula)


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


def test_replace_1():
    const = as_expression(12.0)
    formula = const * 3 + 5.0
    assert formula() == approx(12.0 * 3 + 5.0)
    formula = formula.replace({const: as_expression(13.0)})
    assert formula() == approx(13.0 * 3 + 5.0)


def test_replace_2():
    net = ExternalFunction("net", ("x",))
    x = ExternalVariable("x")
    protected = x[28] >= 1.0
    qualified = (x[12] >= 1.0) | (x[40] >= 1.0)
    hire = net[0] >= net[1]
    prob1 = prob(hire, condition=protected)
    prob2 = prob(hire, condition=~protected)
    formula_1 = prob1 / prob2 >= 0.8
    prob3 = prob(hire, condition=protected & qualified)
    prob4 = prob(hire, condition=~protected & qualified)
    formula_2 = prob3 / prob4 >= 0.8
    formula_3 = formula_1 | formula_2

    subst = {
        prob1: as_expression(0.5),
        prob2: as_expression(0.8),
        prob3: as_expression(0.35),
        prob4: as_expression(0.4),
    }
    formula_1 = formula_1.replace(subst)
    assert not formula_1()
    formula_2 = formula_2.replace(subst)
    assert formula_2()
    formula_3 = formula_3.replace(subst)
    assert formula_3()


def test_propagate_bounds_1():
    x = ExternalVariable("x")
    y = ExternalVariable("y")

    assert x.propagate_bounds(x=(-1.0, 1.0)) == (-1.0, 1.0)
    y_lb = torch.tensor([-1.0, 0.0, 1.0])
    y_ub = torch.tensor([1.0, 0.0, 2.0])
    assert y.propagate_bounds(y=(y_lb, y_ub)) == (y_lb, y_ub)

    ineq1 = x >= y
    ineq2 = x < y

    x_lb = torch.tensor([1.5, -1.0, 0.5])
    x_ub = torch.tensor([2.0, -0.5, 1.5])
    ineq1_res = ineq1.propagate_bounds(x=(x_lb, x_ub), y=(y_lb, y_ub))
    assert ineq1_res[0] == TL.TRUE
    assert ineq1_res[1] == TL.FALSE
    assert ineq1_res[2] == TL.UNKNOWN
    ineq2_res = ineq2.propagate_bounds(x=(x_lb, x_ub), y=(y_lb, y_ub))
    assert ineq2_res[0] == TL.FALSE
    assert ineq2_res[1] == TL.TRUE
    assert ineq2_res[2] == TL.UNKNOWN

    formula = ineq1 | ineq2
    formula_res = formula.propagate_bounds(x=(x_lb, x_ub), y=(y_lb, y_ub))
    assert formula_res[0] == TL.TRUE
    assert formula_res[1] == TL.TRUE
    assert formula_res[2] == TL.UNKNOWN


def test_propagate_bounds_2():
    x = ExternalVariable("x")
    y = ExternalVariable("y")
    z = x * x + 7 * (x * y) - y / (2 * x - 1)

    torch.manual_seed(556917027411149)

    def random_bounds():
        mid = 200 * torch.rand(10) - 100
        dev = 100 * torch.rand(10)
        return mid - dev, mid + dev

    for _ in range(10):
        x_lb, x_ub = random_bounds()
        y_lb, y_ub = random_bounds()
        z_lb, z_ub = z.propagate_bounds(x=(x_lb, x_ub), y=(y_lb, y_ub))

        x_values = x_lb + (x_ub - x_lb) * torch.rand(100, 10)
        y_values = y_lb + (y_ub - y_lb) * torch.rand(100, 10)
        if torch.any(x_lb > x_values) or torch.any(x_ub < x_values):
            raise ValueError()
        if torch.any(y_lb > y_values) or torch.any(y_ub < y_values):
            raise ValueError()

        z_values = z(x=x_values, y=y_values)
        assert torch.all(z_lb <= z_values)
        assert torch.all(z_ub >= z_values)


def test_propagate_bounds_3():
    x = ExternalVariable("x")
    y = ExternalVariable("y")
    z = ExternalVariable("z")
    a = ExternalVariable("a")
    f = max_expr(min_expr(x, y), -z, a * 2)

    def random_bounds():
        mid = 200 * torch.rand(10) - 100
        dev = 100 * torch.rand(10)
        return mid - dev, mid + dev

    for _ in range(10):
        x_lb, x_ub = random_bounds()
        y_lb, y_ub = random_bounds()
        z_lb, z_ub = random_bounds()
        a_lb, a_ub = random_bounds()
        f_lb, f_ub = f.propagate_bounds(
            x=(x_lb, x_ub), y=(y_lb, y_ub), z=(z_lb, z_ub), a=(a_lb, a_ub)
        )

        x_values = x_lb + (x_ub - x_lb) * torch.rand(100, 10)
        y_values = y_lb + (y_ub - y_lb) * torch.rand(100, 10)
        z_values = z_lb + (z_ub - z_lb) * torch.rand(100, 10)
        a_values = a_lb + (a_ub - a_lb) * torch.rand(100, 10)

        f_values = f(x=x_values, y=y_values, z=z_values, a=a_values)
        assert torch.all(f_lb <= f_values)
        assert torch.all(f_ub >= f_values)


if __name__ == "__main__":
    pytest.main()
