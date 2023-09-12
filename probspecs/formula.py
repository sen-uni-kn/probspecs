# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from abc import ABC
from enum import Enum, auto, unique
from dataclasses import dataclass
import operator as ops
from math import prod
from typing import Union

import torch

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
        operands_eval = (operand(**kwargs) for operand in self.operands)
        eval_op = self.op.eval_func()
        return eval_op(operands_eval)

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
        lhs_eval = self.lhs(**kwargs)
        rhs_eval = self.rhs(**kwargs)
        eval_op = self.op.eval_func()
        return eval_op(lhs_eval, rhs_eval)

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
        args_eval = (arg_fn(**kwargs) for arg_fn in self.args)
        eval_op = self.op.eval_func()
        return eval_op(args_eval)

    def __le__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> Inequality:
        other = as_expression(other)
        return Inequality(self, Inequality.Operator.LESS_EQUAL, other)

    def __ge__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> Inequality:
        other = as_expression(other)
        return Inequality(self, Inequality.Operator.GREATER_EQUAL, other)

    def __lt__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> Inequality:
        other = as_expression(other)
        return Inequality(self, Inequality.Operator.LESS_THAN, other)

    def __gt__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> Inequality:
        other = as_expression(other)
        return Inequality(self, Inequality.Operator.GREATER_THAN, other)

    def __eq__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> Formula:
        other = as_expression(other)
        return (self >= other) and (self <= other)

    def __ne__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> Formula:
        other = as_expression(other)
        return (self > other) or (self < other)

    def __add__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.ADD, (self, other))

    def __radd__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.ADD, (other, self))

    def __sub__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.SUBTRACT, (self, other))

    def __rsub__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.SUBTRACT, (other, self))

    def __mul__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.MULTIPLY, (self, other))

    def __rmul__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.MULTIPLY, (other, self))

    def __truediv__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.DIVIDE, (self, other))

    def __rtruediv__(
        self, other: Union["Expression", "Function", torch.Tensor, float]
    ) -> "Expression":
        other = as_expression(other)
        return Expression(Expression.Operator.DIVIDE, (other, self))

    def __getitem__(
        self, item: int | tuple[int, ...] | slice | tuple[slice, ...]
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
        Evaluates the function. Returns the value of the function and
        the remaining positional arguments that this function didn't consume.
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

    def __call__(self, **kwargs) -> torch.Tensor | float:
        return self.val

    def __repr__(self):
        return f"{self.val}"


def as_expression(value: Expression | torch.Tensor | float) -> Expression | Constant:
    """
    Wraps value in a :class:`Constant` if it isn't already
    an expression.
    """
    if not isinstance(value, Expression):
        return Constant(value)
    else:
        return value


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

        :return: The estimate of this probability and the remaining positional
         arguments (a scalar :class:`torch.Tensor`).
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

    def __repr__(self):
        if self.condition is not None:
            return f"P[{self.subject} | {self.condition}]"
        else:
            return f"P[{self.subject}]"


def prob(subject: Formula | Inequality, condition: Formula | Inequality | None = None):
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

    def __call__(self, **kwargs):
        return kwargs[self.name]

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
        wrapper(net=actual_network, x=torch.rand(100, 10))

    """

    func_name: str
    arg_names: tuple[str, ...]

    def __call__(self, **kwargs):
        func = kwargs[self.func_name]
        func_args = (kwargs[name] for name in self.arg_names)
        return func(*func_args)

    def __repr__(self):
        return f"{self.func_name}(" + ", ".join(self.arg_names) + ")"
