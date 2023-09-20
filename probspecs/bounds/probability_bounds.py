# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Generator, Literal
from itertools import groupby

import torch

from ..formula import Probability, ExternalVariable, Constant, Inequality, Formula
from ..verifier import collect_requires_bounds
from .auto_lirpa_params import AutoLiRPAParams


def probability_bounds(
    probability: Probability,
    variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 128,
    auto_lirpa_params: AutoLiRPAParams = AutoLiRPAParams(),
    split_heuristic: Literal["IBP", "longest-edge"] = "IBP",
) -> Generator[tuple[float, float], None, None]:
    """
    Computes a sequence of refined bounds on a probability.
    With each yield of this generator, the lower and upper bounds that it
    produces improve, meaning that the lower bound increases while the upper
    bound decreases.

    To refine the previously computed bounds, :code:`probability_bounds` performs
    branch and bound with input splitting.

    :param probability: The probability to compute bounds on.
    :param variable_bounds: Bounds on all external variables appearing
     in the probability.
    :param batch_size: The number of branches to consider at a time.
    :param auto_lirpa_params: Parameters for running auto_LiRPA.
    :param split_heuristic: Which heuristic to use for selecting dimensions to split.
    :return: A generator that yields improving lower and upper bounds.
    """
    requires_bounds = collect_requires_bounds(probability.subject)

    # make sure probability doesn't contain nested probabilities.
    for term in requires_bounds:
        if isinstance(term, Probability):
            raise ValueError(
                "Computing bounds on probabilities with"
                "nested probabilities is unsupported. "
                f"Found probability {term} in {probability}."
            )

    # If probability.condition has a simple structure, apply it to the
    # variable bounds directly.
    probability, variable_bounds = apply_simple_conditions(probability, variable_bounds)

    if probability.condition is not None:
        # use P(A|B) = P(A AND B) / P(B) to bound conditional probabilities.
        p_conjoined = Probability(probability.subject & probability.condition)
        p_condition = Probability(probability.condition)
        p_conjoined_bounds = probability_bounds(
            p_conjoined, variable_bounds, batch_size, auto_lirpa_params, split_heuristic
        )
        p_condition_bounds = probability_bounds(
            p_condition, variable_bounds, batch_size, auto_lirpa_params, split_heuristic
        )

        def conditional_bounds_gen():
            prob_formula = ExternalVariable("conj") / ExternalVariable("cond")
            for conj_bounds, cond_bounds in zip(p_conjoined_bounds, p_condition_bounds):
                prob_bounds = prob_formula.propagate_bounds(
                    conj=conj_bounds, cond=cond_bounds
                )
                yield prob_bounds

        yield from conditional_bounds_gen()

    # TODO:


def apply_simple_conditions(
    probability: Probability,
    variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[Probability, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """
    If :code:`probability.condition` has the structure
    :code:`x >= y AND ...`,
    where :code:`x` is an :class:`ExternalVariable` and
    :code:`y` is a :class:`Constant`,
    this function directly applies the condition to a set of bounds
    on the external variables in :code:`probability`.
    If the condition contains further terms, they are retained.
    This function can also apply multiple simple inequalities, including
    inequalities like :code:`y <= x`.

    :param probability: The :class:`Probability` whose condition to apply
     to the variable bounds.
    :param variable_bounds: Bounds on all external variables appearing in
     :code:`probability`.
    :return: The probability with the simplified condition (conditions
     :code:`x >= y` removed) and the new variable bounds where the removed
     conditions were applied directly.
    """

    def is_simple(term: Formula | Inequality) -> bool:
        return (
            isinstance(term, Inequality)
            and (
                (
                    isinstance(term.lhs, ExternalVariable)
                    and isinstance(term.rhs, Constant)
                )
                or (
                    isinstance(term.rhs, ExternalVariable)
                    and isinstance(term.lhs, Constant)
                )
            )
            and term.op
            in (Inequality.Operator.LESS_EQUAL, Inequality.Operator.GREATER_EQUAL)
        )

    cond = probability.condition
    if isinstance(cond, Formula) and cond.op is Formula.Operator.AND:
        top_level_conjuncts = list(cond.operands)
        single_conds = []
        remaining_conds = []
        # This AND may contain nested ANDs
        while len(top_level_conjuncts) > 0:
            conj = top_level_conjuncts.pop()
            terms = conj.operands
            for term in terms:
                if is_simple(term):
                    single_conds.append(term)
                elif isinstance(term, Formula) and term.op is Formula.Operator.AND:
                    top_level_conjuncts += term.operands
                else:
                    remaining_conds.append(term)
    elif is_simple(cond):
        single_conds = [cond]
        remaining_conds = []
    else:
        return probability, variable_bounds

    new_bounds = dict(variable_bounds)
    for ineq in single_conds:
        var = ineq.lhs if isinstance(ineq.lhs, ExternalVariable) else ineq.rhs
        const = ineq.lhs if isinstance(ineq.lhs, Constant) else ineq.rhs

        lb, ub = variable_bounds[var.name]
        if ineq.op is Inequality.Operator.LESS_EQUAL:
            lb = torch.clamp(lb, max=const.val)
            ub = torch.clamp(ub, max=const.val)
        else:  # var >= const
            lb = torch.clamp(lb, min=const.val)
            ub = torch.clamp(ub, min=const.val)
        new_bounds[var.name] = (lb, ub)

    if len(remaining_conds) == 0:
        probability = Probability(probability.subject)
    else:
        remaining_conds = Formula(Formula.Operator.AND, tuple(remaining_conds))
        probability = Probability(probability.subject, remaining_conds)
    return probability, new_bounds
