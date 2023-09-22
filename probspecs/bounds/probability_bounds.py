# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from math import prod
from typing import Generator, Literal

import torch
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor

from .auto_lirpa_params import AutoLiRPAParams
from .branch_store import BranchStore
from ..formula import (
    ExternalFunction,
    Function,
    Probability,
    ExternalVariable,
    Constant,
    Inequality,
    Formula,
)
from ..population_model import ProbabilityDistribution
from ..verifier import collect_requires_bounds
from ..input_space import InputSpace
from .utils import construct_bounded_tensor


def probability_bounds(
    probability: Probability,
    networks: dict[str, nn.Module],
    variable_domains: dict[str, InputSpace],
    variable_distributions: dict[str, ProbabilityDistribution],
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
    :param networks: The neural networks appearing as :class:`ExternalFunction`
     objects in :code:`probability`.
    :param variable_domains: :class:`InputSpace` objects for all external variables
     (:class:`ExternalVariable` objects and arguments of
     :class:`ExternalFunction` objects)
     appearing in :code:`probability`.
    :param variable_distributions: Probability distributions for all external variables
     appearing in :code:`probability` (see :code:`variable_domains`).
    :param batch_size: The number of branches to consider at a time.
    :param auto_lirpa_params: Parameters for running auto_LiRPA.
    :param split_heuristic: Which heuristic to use for selecting dimensions to split.
    :return: A generator that yields improving lower and upper bounds.
    """
    if probability.condition is not None:
        # use P(A|B) = P(A AND B) / P(B) to bound conditional probabilities.
        p_conjoined = Probability(probability.subject & probability.condition)
        p_condition = Probability(probability.condition)
        p_conjoined_bounds = probability_bounds(
            p_conjoined,
            networks,
            variable_domains,
            variable_distributions,
            batch_size,
            auto_lirpa_params,
            split_heuristic,
        )
        p_condition_bounds = probability_bounds(
            p_condition,
            networks,
            variable_domains,
            variable_distributions,
            batch_size,
            auto_lirpa_params,
            split_heuristic,
        )

        def conditional_bounds_gen():
            prob_formula = ExternalVariable("conj") / ExternalVariable("cond")
            for conj_bounds, cond_bounds in zip(p_conjoined_bounds, p_condition_bounds):
                prob_bounds = prob_formula.propagate_bounds(
                    conj=conj_bounds, cond=cond_bounds
                )
                yield prob_bounds

        yield from conditional_bounds_gen()

    variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        var: domain.input_bounds for var, domain in variable_domains.items()
    }

    def eval_bounds(
        term_: Function, var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(term_, ExternalVariable):
            return term_.propagate_bounds(**var_bounds)
        elif isinstance(term_, ExternalFunction):
            network = networks[term_.func_name]
            bounded_tensor = construct_bounded_tensor(*var_bounds[term_.arg_names[0]])
            return network.compute_bounds(
                x=(bounded_tensor,), method=auto_lirpa_params.method
            )
        else:
            raise NotImplementedError()

    def probability_mass(var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        return prod(
            variable_distributions[var].cdf(var_ubs)
            - variable_distributions[var].cdf(var_lbs)
            for var, (var_lbs, var_ubs) in var_bounds.items()
        )

    # simplify the probability subject
    subj, variable_bounds = apply_symbolic_bounds(probability.subject, variable_bounds)

    requires_bounds = collect_requires_bounds(subj)
    bounded_terms_substitution = {
        term: ExternalVariable(str(term)) for term in requires_bounds
    }
    top_level_subj = subj.replace(bounded_terms_substitution)

    # make sure probability doesn't contain nested probabilities.
    for term in requires_bounds:
        if isinstance(term, Probability):
            raise ValueError(
                "Computing bounds on probabilities with"
                "nested probabilities is unsupported. "
                f"Found probability {term} in {probability}."
            )

    # auto_LiRPA: no networks with multiple inputs
    for external in requires_bounds:
        if isinstance(external, ExternalFunction):
            if len(external.arg_names) > 1:
                raise ValueError(
                    f"ExternalFunctions with multiple inputs are unsupported "
                    f"by probability_bounds. "
                    f"Found function with multiple arguments: {external}."
                )

    networks = {
        external.func_name: BoundedModule(
            networks[external.func_name],
            variable_bounds[external.arg_names[0]][0],  # lower bound of arg
            auto_lirpa_params.bound_ops,
        )
        for external in requires_bounds
        if isinstance(external, ExternalFunction)
    }

    branches = BranchStore(
        in_shape=None,  # input shapes stored separately for each variable
        **{
            f"{var}_lbs": domain.input_shape for var, domain in variable_domains.items()
        },
        **{
            f"{var}_ubs": domain.input_shape for var, domain in variable_domains.items()
        },
        out_shape=(1,),  # a satisfaction score
        probability_mass=(1,),
    )
    intermediate_bounds = {
        str(term): eval_bounds(term, variable_bounds) for term in requires_bounds
    }
    # TODO: need bounds on sat_score
    sat_score = top_level_subj.satisfaction_function(**intermediate_bounds)
    prob_mass = probability_mass(variable_bounds)
    branches.append(
        **{
            f"{var}_lbs": domain.input_shape for var, domain in variable_domains.items()
        },
        **{
            f"{var}_ubs": domain.input_shape for var, domain in variable_domains.items()
        },
        out_shape=(1,),  # a satisfaction score
        probability_mass=(1,),
    )

    def eval_probability_bounds() -> tuple[torch.Tensor, torch.Tensor]:
        pass  # TODO

    # TODO: bound probability (this is not how it works)
    best_lb, best_ub = eval_probability_bounds()
    yield (best_lb, best_ub)

    if len(networks) == 0:
        # No need to refine bounds in this case
        while True:
            yield (best_lb, best_ub)

    # TODO: refine branches


def apply_symbolic_bounds(
    formula: Formula | Inequality,
    variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[Formula, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """
    If :code:`formula` has the structure
    :code:`x >= y AND ...`,
    where :code:`x` is an :class:`ExternalVariable` and
    :code:`y` is a :class:`Constant`,
    the term :code:`x >= y` can be simplified when the values
    for which the formula is evaluated are bounded.
    Concretely, when we have bounds on the external variable :code:`x`,
    we can removed the term :code:`x >= y` from the formula
    applying :code:`x >= y` directly to the bounds on :code:`x`.
    This function performs this simplification.
    This function can also apply multiple simple inequalities, including
    inequalities like :code:`y <= x`.

    :param formula: The :class:`Formula` to simplify.
     For convenience, this function als accepts :class:`Inequality` objects.
     When an inequality can be simplified, the formula that this function returns
     is equivalent to :code:`True`.
    :param variable_bounds: Bounds on all external variables appearing in
     :code:`formula`.
    :return: The simplified formula and the new variable bounds
     where the term removed from the formula were applied directly.
    """

    def is_simple(term_: Formula | Inequality) -> bool:
        return (
            isinstance(term_, Inequality)
            and (
                (
                    isinstance(term_.lhs, ExternalVariable)
                    and isinstance(term_.rhs, Constant)
                )
                or (
                    isinstance(term_.rhs, ExternalVariable)
                    and isinstance(term_.lhs, Constant)
                )
            )
            and term_.op
            in (Inequality.Operator.LESS_EQUAL, Inequality.Operator.GREATER_EQUAL)
        )

    if isinstance(formula, Formula) and formula.op is Formula.Operator.AND:
        top_level_conjuncts = list(formula.operands)
        simple_terms = []
        remaining_terms = []
        # This AND may contain nested ANDs
        while len(top_level_conjuncts) > 0:
            conj = top_level_conjuncts.pop()
            terms = conj.operands
            for term in terms:
                if is_simple(term):
                    simple_terms.append(term)
                elif isinstance(term, Formula) and term.op is Formula.Operator.AND:
                    top_level_conjuncts += term.operands
                else:
                    remaining_terms.append(term)
    elif is_simple(formula):
        simple_terms = [formula]
        remaining_terms = []
    else:
        return formula, variable_bounds

    new_bounds = dict(variable_bounds)
    for ineq in simple_terms:
        var = ineq.lhs if isinstance(ineq.lhs, ExternalVariable) else ineq.rhs
        const = ineq.lhs if isinstance(ineq.lhs, Constant) else ineq.rhs

        lb, ub = variable_bounds[var.name]
        if ineq.op is Inequality.Operator.LESS_EQUAL:
            ub = torch.clamp(ub, max=const.val)
        else:  # var >= const
            lb = torch.clamp(lb, min=const.val)
        new_bounds[var.name] = (lb, ub)

    remaining_terms = Formula(Formula.Operator.AND, tuple(remaining_terms))
    return remaining_terms, new_bounds
