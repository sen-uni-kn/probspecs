# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import itertools
from abc import ABC
from enum import Enum, auto, unique
from functools import reduce
from dataclasses import dataclass
import operator as ops
from math import prod
from typing import Union

import numpy as np
import torch

from .trinary_logic import TrinaryLogic as TL
from .utils import contains_unbracketed


__all__ = [
    "Formula",
    "Inequality",
    "Expression",
    "Function",
    "Constant",
    "ElementAccess",
    "Probability",
    "ExternalVariable",
    "ExternalFunction",
    "as_expression",
    "prob",
]


MAIN_TYPES = Union["Formula", "Inequality", "Expression", "Function"]
PRECOMPUTED = dict[MAIN_TYPES, torch.Tensor | bool]


@dataclass(frozen=True)
class Formula:
    """
    A formula :math:`P(x) \\wedge R(x)`,
    :math:`P(x) \\vee R(x)` or :math:`\\neg P(x)`
    where :math:`P` and :math:`R` are Inequalities.

    For negating formulas, use :code:`~formula`.
    Similarly, use :code:`&`
    and :code:`|` to and- or or- connect formulae.
    """

    @unique
    class Operator(Enum):
        AND = auto()
        OR = auto()
        NOT = auto()

        def eval_func(self):
            match self:
                case self.AND:
                    return all
                case self.OR:
                    return any
                case self.NOT:
                    return ops.not_
                case _:
                    raise NotImplementedError()

        def __str__(self):
            match self:
                case self.NOT:
                    return "¬"
                case self.AND:
                    return "∧"
                case self.OR:
                    return "∨"
                case _:
                    raise NotImplementedError()

    op: Operator
    operands: tuple[Union["Formula", "Inequality"], ...]

    def __post_init__(self):
        if self.op == self.Operator.NOT and len(self.operands) != 1:
            raise ValueError("NOT requires exactly one argument.")

    def __call__(self, **kwargs) -> torch.Tensor | bool:
        """
        Evaluates this formula.

        :param kwargs: External functions and values of variables
         for evaluating :class:`ExternalFunction` and :class:`ExternalVariable`
         objects.
        :return: The value of this formula, as a boolean or a tensor
         of booleans.
        """
        operands_eval = (operand(**kwargs) for operand in self.operands)
        eval_op = self.op.eval_func()
        return eval_op(operands_eval)

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> np.ndarray[tuple, TL] | TL:
        """
        Propagates bounds on external variables and external function through
        the formula, evaluating whether the formula certainly holds or
        is certainly violated given the bounds.

        :param bounds: The bounds on the :class:`ExternalVariable`
         and :class:`ExternalFunction` objects in this formula.
        :return: :code:`TrinaryLogic.TRUE` if this formula certainly holds,
         :code:`TrinaryLogic.FALSE` if it's certainly violated,
         and :code:`TrinaryLogic.UNKNOWN` otherwise.
        """
        child_results = (child.propagate_bounds(**bounds) for child in self.operands)
        match self.op:
            case self.Operator.NOT:
                return TL.not_(next(child_results))
            case self.Operator.AND:
                return TL.and_(*child_results)
            case self.Operator.OR:
                return TL.or_(*child_results)
            case _:
                raise NotImplementedError()

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> "Formula":
        """
        Replaces formulae, inequalities, expressions, and functions
        inside this formula.

        :param substitutions: Which formula, inequalities, expressions,
         and functions to replace by what.
        :return: A new formula with the substitutions applied.
        """
        if self in substitutions:
            return substitutions[self]
        else:
            return Formula(
                self.op, tuple(child.replace(substitutions) for child in self.operands)
            )

    def __and__(self, other: Union["Formula", "Inequality"]) -> "Formula":
        return Formula(Formula.Operator.AND, (self, other))

    def __or__(self, other: Union["Formula", "Inequality"]) -> "Formula":
        return Formula(Formula.Operator.OR, (self, other))

    def __invert__(self) -> "Formula":
        return Formula(Formula.Operator.NOT, (self,))

    def __repr__(self):
        if self.op == self.Operator.NOT:
            return f"{self.op}({self.operands[0]})"

        def convert_operand(operand):
            res = str(operand)
            if self.op == self.Operator.AND and contains_unbracketed(
                res, (str(self.Operator.OR),)
            ):
                return "(" + res + ")"
            else:
                return res

        return f" {self.op} ".join(
            [convert_operand(operand) for operand in self.operands]
        )


@dataclass(frozen=True)
class Inequality:
    """
    An inequality :math:`f(x) \\ otherlesseqgtr b`
    where :math:`\\lesseqgtr` is either :math:`\\leq`,
    :math:`\\geq`, or :math:`<`.

    You can combine Inequalities into formulas
    using :code:`~`, :code:`&`, and :code:`|`.
    """

    @unique
    class Operator(Enum):
        LESS_EQUAL = auto()
        GREATER_EQUAL = auto()
        LESS_THAN = auto()
        GREATER_THAN = auto()

        def eval_func(self):
            match self:
                case self.LESS_EQUAL:
                    return ops.le
                case self.GREATER_EQUAL:
                    return ops.ge
                case self.LESS_THAN:
                    return ops.lt
                case self.GREATER_THAN:
                    return ops.gt
                case _:
                    raise NotImplementedError()

        def __str__(self):
            match self:
                case self.LESS_EQUAL:
                    return "≤"
                case self.GREATER_EQUAL:
                    return "≥"
                case self.LESS_THAN:
                    return "<"
                case self.GREATER_THAN:
                    return ">"
                case _:
                    raise NotImplementedError()

    lhs: "Expression"
    op: Operator
    rhs: "Expression"

    def __call__(self, **kwargs) -> torch.Tensor | bool:
        """
        Evaluates this inequality.

        :param kwargs: External functions and values of variables
         for evaluating :class:`ExternalFunction` and :class:`ExternalVariable`
         objects.
        :return: The value of this inequality, as a boolean or a tensor
         of booleans.
        """
        lhs_eval = self.lhs(**kwargs)
        rhs_eval = self.rhs(**kwargs)
        eval_op = self.op.eval_func()
        return eval_op(lhs_eval, rhs_eval)

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> np.ndarray[tuple, TL] | TL:
        """
        Propagates bounds on external variables and external function through
        the inequality, evaluating whether the inequality certainly holds or
        is certainly violated given the bounds.

        :param bounds: The bounds on the :class:`ExternalVariable`
         and :class:`ExternalFunction` objects in this formula.
        :return: :code:`TrinaryLogic.TRUE` if this inequality certainly holds,
         :code:`TrinaryLogic.FALSE` if it's certainly violated,
         and :code:`TrinaryLogic.UNKNOWN` otherwise.
        """
        lhs_lb, lhs_ub = self.lhs.propagate_bounds(**bounds)
        rhs_lb, rhs_ub = self.rhs.propagate_bounds(**bounds)

        if self.op in (self.Operator.GREATER_EQUAL, self.Operator.GREATER_THAN):
            # turn >= into <= by switching the sides of the inequality.
            lhs_lb, rhs_lb = rhs_lb, lhs_lb
            lhs_ub, rhs_ub = rhs_ub, lhs_ub

        if self.op in (self.Operator.LESS_THAN, self.Operator.GREATER_THAN):
            compare = ops.lt
            anti = ops.ge
        else:
            compare = ops.le
            anti = ops.gt

        return np.where(
            compare(lhs_ub, rhs_lb),
            TL.TRUE,
            np.where(anti(lhs_lb, rhs_ub), TL.FALSE, TL.UNKNOWN),
        )

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> "Inequality":
        """
        Replaces formulae, inequalities, expressions, and functions
        inside this inequality.

        :param substitutions: Which formula, inequalities, expressions,
         and functions to replace by what.
        :return: A new inequality with the substitutions applied.
        """
        if self in substitutions:
            return substitutions[self]
        else:
            return Inequality(
                self.lhs.replace(substitutions),
                self.op,
                self.rhs.replace(substitutions),
            )

    def __and__(self, other: Union[Formula, "Inequality"]) -> "Formula":
        return Formula(Formula.Operator.AND, (self, other))

    def __or__(self, other: Union[Formula, "Inequality"]) -> "Formula":
        return Formula(Formula.Operator.OR, (self, other))

    def __invert__(self) -> Formula:
        return Formula(Formula.Operator.NOT, (self,))

    def __repr__(self):
        return f"{self.lhs} {self.op} {self.rhs}"


@dataclass(frozen=True)
class Expression:
    @unique
    class Operator(Enum):
        ADD = auto()
        SUBTRACT = auto()
        MULTIPLY = auto()
        DIVIDE = auto()

        def eval_func(self):
            match self:
                case self.ADD:
                    return sum
                case self.SUBTRACT:
                    return lambda xs: xs[0] - sum(xs[1:])
                case self.MULTIPLY:
                    return prod
                case self.DIVIDE:
                    return lambda xs: ops.truediv(xs[0], xs[1])
                case _:
                    raise NotImplementedError()

        def __str__(self):
            match self:
                case self.ADD:
                    return "+"
                case self.SUBTRACT:
                    return "-"
                case self.MULTIPLY:
                    return "*"
                case self.DIVIDE:
                    return "/"
                case _:
                    raise NotImplementedError()

    op: Operator
    args: tuple[Union["Expression", "Function"], ...]

    def __post_init__(self):
        if len(self.args) < 2:
            raise ValueError("Expressions require at least two arguments.")
        if self.op == self.Operator.DIVIDE and len(self.args) != 2:
            raise ValueError("DIVIDE requires exactly two arguments.")

    def __call__(self, **kwargs) -> torch.Tensor | float:
        """
        Evaluates this expression.

        :param kwargs: External functions and values of variables
         for evaluating :class:`ExternalFunction` and :class:`ExternalVariable`
         objects.
        :return: The value of this expression, as a float or a tensor.
        """
        args_eval = tuple(arg_fn(**kwargs) for arg_fn in self.args)
        eval_op = self.op.eval_func()
        return eval_op(args_eval)

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """
        Propagates bounds on external variables and external function through
        this expression, producing bounds on the expression itself.

        :param bounds: The bounds on the :class:`ExternalVariable`
         and :class:`ExternalFunction` objects in this formula.
        :return: The lower bound and the upper bound of this expression
         derived from the bounds.
        """
        arg_bounds = [arg.propagate_bounds(**bounds) for arg in self.args]
        arg_lbs, arg_ubs = zip(*arg_bounds)
        match self.op:
            case self.Operator.ADD:
                return sum(arg_lbs), sum(arg_ubs)
            case self.Operator.SUBTRACT:
                eval_op = self.op.eval_func()
                lb = eval_op((arg_lbs[0], *arg_ubs[1:]))
                ub = eval_op((arg_ubs[0], *arg_lbs[1:]))
                return lb, ub
            case self.Operator.MULTIPLY:
                # for x*y compute x_lb * y_lb, x_lb * y_ub, x_ub * y_lb and x_ub * y_ub
                all_combinations = itertools.product(*arg_bounds)
                multiplied = [prod(combination) for combination in all_combinations]
                multiplied = [torch.as_tensor(res) for res in multiplied]
                lb = reduce(  # if min(multiplied) only worked for tensors
                    torch.minimum,
                    multiplied,
                    torch.full_like(multiplied[0], torch.inf, dtype=torch.float),
                )
                ub = reduce(
                    torch.maximum,
                    multiplied,
                    torch.full_like(multiplied[0], -torch.inf, dtype=torch.float),
                )
                return lb, ub
            case self.Operator.DIVIDE:
                (x_lb, x_ub), (y_lb, y_ub) = arg_bounds
                x_lb, x_ub = torch.as_tensor(x_lb), torch.as_tensor(x_ub)
                y_lb, y_ub = torch.as_tensor(y_lb), torch.as_tensor(y_ub)

                lb = torch.where(
                    y_lb.gt(0) | (y_lb.eq(0) & y_ub.gt(0)), x_lb * 1 / y_ub, -torch.inf
                )
                ub = torch.where(y_lb.gt(0) | y_ub.eq(0), x_ub * 1 / y_lb, torch.inf)
                return lb, ub
            case _:
                raise NotImplementedError()

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> "Expression":
        """
        Replaces formulae, inequalities, expressions, and functions
        inside this expression.

        :param substitutions: Which formula, inequalities, expressions,
         and functions to replace by what.
        :return: A new expression with the substitutions applied.
        """
        if self in substitutions:
            return substitutions[self]
        else:
            return Expression(
                self.op, tuple(child.replace(substitutions) for child in self.args)
            )

    def __le__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> Inequality:
        other = as_expression(other)
        return Inequality(self, Inequality.Operator.LESS_EQUAL, other)

    def __ge__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> Inequality:
        other = as_expression(other)
        return Inequality(self, Inequality.Operator.GREATER_EQUAL, other)

    def __lt__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> Inequality:
        other = as_expression(other)
        return Inequality(self, Inequality.Operator.LESS_THAN, other)

    def __gt__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> Inequality:
        other = as_expression(other)
        return Inequality(self, Inequality.Operator.GREATER_THAN, other)

    def __eq__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> Formula:
        other = as_expression(other)
        return (self >= other) & (self <= other)

    def __ne__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> Formula:
        other = as_expression(other)
        return (self > other) | (self < other)

    def __add__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.ADD, (self, other))

    def __radd__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.ADD, (other, self))

    def __sub__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.SUBTRACT, (self, other))

    def __rsub__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.SUBTRACT, (other, self))

    def __mul__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.MULTIPLY, (self, other))

    def __rmul__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.MULTIPLY, (other, self))

    def __truediv__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.DIVIDE, (self, other))

    def __rtruediv__(
        self: Union["Expression", "Function"],
        other: Union["Expression", "Function", torch.Tensor, float],
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.DIVIDE, (other, self))

    def __getitem__(
        self: Union["Expression", "Function"],
        item: int | tuple[int, ...] | slice | tuple[slice, ...],
    ) -> "Function":
        return ElementAccess(self, item)

    def __repr__(self):
        def convert_arg(operand):
            res = str(operand)
            if self.op in (
                self.Operator.MULTIPLY,
                self.Operator.DIVIDE,
            ) and contains_unbracketed(
                res, (str(self.Operator.ADD), str(self.Operator.SUBTRACT))
            ):
                return "(" + res + ")"
            else:
                return res

        return f" {self.op} ".join([convert_arg(arg) for arg in self.args])


class Function(ABC):
    def __call__(self, **kwargs) -> torch.Tensor | float:
        """
        Evaluates this function.

        :param kwargs: External functions and values of variables
         for evaluating :class:`ExternalFunction` and :class:`ExternalVariable`
         objects.
        :return: The value of this function, as a float or a tensor.
        """
        raise NotImplementedError()

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """
        Propagates bounds on external variables and external function through
        this function, producing bounds on the function itself.

        :param bounds: The bounds on the :class:`ExternalVariable`
         and :class:`ExternalFunction` objects in this formula.
        :return: The lower bound and the upper bound of this function
         derived from the bounds.
        """
        raise NotImplementedError()

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> MAIN_TYPES:
        """
        Replaces formulae, inequalities, expressions, and functions
        inside this function.

        :param substitutions: Which formula, inequalities, expressions,
         and functions to replace by what.
        """
        raise NotImplementedError()

    def __le__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> Inequality:
        return Expression.__le__(self, other)

    def __ge__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> Inequality:
        return Expression.__ge__(self, other)

    def __lt__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> Inequality:
        return Expression.__lt__(self, other)

    def __gt__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> Inequality:
        return Expression.__gt__(self, other)

    def __eq__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> Formula:
        return Expression.__eq__(self, other)

    def __ne__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> Formula:
        return Expression.__ne__(self, other)

    def __add__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> "Expression":
        return Expression.__add__(self, other)

    def __radd__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> "Expression":
        return Expression.__radd__(self, other)

    def __sub__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> "Expression":
        return Expression.__sub__(self, other)

    def __rsub__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> "Expression":
        return Expression.__rsub__(self, other)

    def __mul__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> "Expression":
        return Expression.__mul__(self, other)

    def __rmul__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> "Expression":
        return Expression.__rmul__(self, other)

    def __truediv__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> "Expression":
        return Expression.__truediv__(self, other)

    def __rtruediv__(
        self, other: Union[Expression, "Function", torch.Tensor, float]
    ) -> "Expression":
        return Expression.__rtruediv__(self, other)

    def __getitem__(
        self, item: int | tuple[int, ...] | slice | tuple[slice, ...]
    ) -> "Function":
        return Expression.__getitem__(self, item)


@dataclass(frozen=True)
class Constant(Function):
    val: torch.Tensor | float

    def __call__(
        self, **kwargs
    ) -> torch.Tensor | float:
        """
        Returns the value of this constant.

        :param kwargs: External functions and values of variables
         for evaluating :class:`ExternalFunction` and :class:`ExternalVariable`
         objects.
        """
        return self.val

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        return self.val, self.val

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> MAIN_TYPES:
        return substitutions.get(self, self)

    def __repr__(self):
        return f"{self.val}"


def as_expression(
    value: Expression | Function | torch.Tensor | float | int,
) -> Expression | Constant:
    """
    Wraps value in a :class:`Constant` if it isn't already an expression or formula.
    """
    if isinstance(value, (Expression, Function)):
        return value
    elif isinstance(value, (Formula, Inequality)):
        raise ValueError(f"Can not convert {type(value)} to Expression.")
    else:
        return Constant(value)


@dataclass(frozen=True)
class ElementAccess(Function):
    """
    Access an element of the result of an expression or function.
    """

    source: Expression | Function
    target_index: int | tuple[int, ...] | slice | tuple[slice, ...]

    def __call__(self, **kwargs) -> torch.Tensor | float:
        source_val = self.source(**kwargs)
        return source_val[self.target_index]

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        source_lb, source_ub = self.source(**bounds)
        return source_lb[self.target_index], source_ub[self.target_index]

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> MAIN_TYPES:
        if self in substitutions:
            return substitutions[self]
        else:
            return ElementAccess(self.source.replace(substitutions), self.target_index)

    def __repr__(self):
        source_str = str(self.source)
        if isinstance(self.source, Expression):
            source_str = f"({source_str})"
        return f"{source_str}[{self.target_index}]"


@dataclass(frozen=True)
class Probability(Function):
    """
    A conditional probability of the form :math:`P(subject | condition)`.
    """

    subject: Formula | Inequality
    condition: Formula | Inequality | None = None

    def __call__(self, **kwargs) -> torch.Tensor:
        """
        Empirically estimate this probability.
        This is only a sensible estimate of evaluating the subject formula and
        the condition formula results in a tensor with several arguments.
        Otherwise, the estimate is 1 if the subject formula holds and 0 otherwise.
        The value is undefined (`nan`) if the condition formula never holds.

        :param kwargs: External functions and values of variables
         for evaluating :class:`ExternalFunction` and :class:`ExternalVariable`
         objects.
        :return: An estimate of this probability as scalar :class:`torch.Tensor`.
        """
        subject_vals = self.subject(**kwargs)
        subject_vals = torch.as_tensor(subject_vals)
        if self.condition is not None:
            condition_vals = self.condition(**kwargs)
            condition_vals = torch.as_tensor(condition_vals)
            if condition_vals.float().sum() == 0:  # condition matches nowhere
                return torch.tensor(torch.nan)
            subject_vals = subject_vals[condition_vals]
        return subject_vals.float().mean()

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        raise NotImplementedError()

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> MAIN_TYPES:
        if self in substitutions:
            return substitutions[self]
        else:
            return Probability(
                self.subject.replace(substitutions),
                self.condition.replace(substitutions),
            )

    def _estimate(self, subject_vals, condition_vals):
        subject_vals = torch.as_tensor(subject_vals)
        if self.condition is not None:
            condition_vals = torch.as_tensor(condition_vals)
            if condition_vals.float().sum() == 0:  # condition matches nowhere
                return torch.tensor(torch.nan)
            subject_vals = subject_vals[condition_vals]
        return subject_vals.float().mean()

    def __repr__(self):
        if self.condition is not None:
            return f"P[{self.subject} | {self.condition}]"
        else:
            return f"P[{self.subject}]"


def prob(subject: Formula | Inequality, condition: Formula | Inequality | None = None):
    """
    Creates a :class:`Probability` object.

    :param subject: The subject formula of the probability.
    :param condition: The condition formula of the probability.
    :return: A new :class:`Probability`.
    """
    return Probability(subject, condition)


@dataclass(frozen=True)
class ExternalVariable(Function):
    """
    Wraps an external variable.
    The value of the variable is retrieved a :code:`kwargs` dictionary,
    for example, when evaluating the function using a unique variable name.

    Example usage:

        wrapper = ExternalVariable("x")
        formula = wrapper + 12.0
        formula(x=torch.rand(100, 10))

    """

    name: str

    def __call__(
        self, **kwargs
    ) -> torch.Tensor | float:
        return kwargs[self.name]

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        return bounds[self.name]

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> MAIN_TYPES:
        return substitutions.get(self, self)

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class ExternalFunction(Function):
    """
    Wraps a python function, such as a :code:`torch` neural network.
    The wrapper consists of an identifier of the external function
    (:code:`func_name`) and identifies for the function arguments.
    These identifiers are used for retrieving the function and
    the function arguments from a :code:`kwargs` dictionary,
    for example, when evaluating the function.

    Example usage:

        wrapper = ExternalFunction("net", "x")
        formula = wrapper + 2.0 >= 5.0
        formula(net=actual_network, x=torch.rand(100, 10))

    """

    func_name: str
    arg_names: tuple[str, ...]

    def __call__(self, **kwargs) -> torch.Tensor | float:
        func = kwargs[self.func_name]
        func_args = (kwargs[name] for name in self.arg_names)
        return func(*func_args)

    def propagate_bounds(
        self, **bounds: tuple[torch.Tensor | float, torch.Tensor | float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """
        Tries to retrieve
        `func_name(arg_names[0], ..., arg_names[-1])` from :code:`bounds`
        first, then tries to retrieve `func_name` from :code:´bounds`.
        One of these keys needs to be present in :code:`bounds`.
        """
        with_args_name = self.func_name + "(" + ",".join(self.arg_names) + ")"
        if with_args_name in bounds:
            return bounds[with_args_name]
        elif self.func_name in bounds:
            return bounds[self.func_name]
        else:
            raise ValueError(
                f"The bounds dictionary neither contains bounds"
                f"for {with_args_name}, nor for {self.func_name}."
            )

    def replace(self, substitutions: dict[MAIN_TYPES, MAIN_TYPES]) -> MAIN_TYPES:
        return substitutions.get(self, self)

    def __repr__(self):
        return f"{self.func_name}(" + ", ".join(self.arg_names) + ")"
