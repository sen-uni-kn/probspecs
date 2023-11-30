# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from ..formula import (
    Formula,
    Inequality,
    Expression,
    Function,
    Probability,
    ExternalFunction,
    ExternalVariable,
)


def collect_requires_bounds(
    term: Formula | Inequality | Expression | Function,
) -> tuple[Function, ...]:
    """
    Determine all :class:`Probability`, :class:`ExternalFunction`,
    and :class:`ExternalVariable` objects in the :class:`Formula`,
    :class:`Inequality`, :class:`Expression`, or :class:`Function`
    :code:`term`.
    """

    def is_prob_func_or_variable(term_):
        return isinstance(term_, Probability | ExternalFunction | ExternalVariable)

    return term.collect(is_prob_func_or_variable)
