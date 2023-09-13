# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Callable

import torch

from .formula import (
    Formula,
    Probability,
    Function,
    Inequality,
    Expression,
    ElementAccess,
    ExternalFunction,
    ExternalVariable,
)


@unique
class VerificationStatus(Enum):
    SATISFIED = auto()
    VIOLATED = auto()
    UNKNOWN = auto()
    ERROR = auto()


def verify(
    formula: Formula,
    externals: dict[str, Callable | torch.Tensor],
    externals_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    workers: tuple[str | torch.device, ...],
) -> tuple[VerificationStatus, dict[Function, tuple[torch.Tensor, torch.Tensor]]]:
    """
    Verifies a formula.

    :param formula: The formula to verify.
    :param formula_kwargs: Keyword arguments for evaluating the formula,
     for example neural networks that appear in the formula.
     See :class:`ExternalFunction` and :class:`ExternalVariable`.
    :param workers: The devices (cpu, cuda:1, cuda:2, ...)
     to use for the different workers computing bounds.
     The length of this tuple determines the number of workers.
    :return: The status of verification and a verification witness.
     The verification witness is a set of bounds for functions in :code:`formula`
     that allows to prove/disprove :code:`formula`.
    """

    def collect_requires_bounds(
        obj: Formula | Inequality | Expression | Function,
    ) -> tuple[Function, ...]:
        match obj:
            case Formula(_, children) | Expression(_, children):
                return sum(
                    (collect_requires_bounds(child) for child in children),
                    start=(),
                )
            case Inequality():
                lhs_result = collect_requires_bounds(obj.lhs)
                rhs_result = collect_requires_bounds(obj.rhs)
                return lhs_result + rhs_result
            case ElementAccess():
                return collect_requires_bounds(obj.source)
            case Probability() | ExternalFunction() | ExternalVariable():
                return (obj,)

    requires_bounds = collect_requires_bounds(formula)
    bounded_terms_substitution = {
        term: ExternalVariable(str(term)) for term in requires_bounds
    }
    top_level_formula = formula.replace(bounded_terms_substitution)
