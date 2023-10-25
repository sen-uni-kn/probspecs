# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from collections import OrderedDict
from dataclasses import dataclass
from math import prod
from typing import Generator, Literal, Sequence, Callable

import torch
from torch import nn
from auto_LiRPA import BoundedModule

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
from ..probability_distribution import ProbabilityDistribution
from ..verifier import collect_requires_bounds
from ..input_space import (
    InputSpace,
    TensorInputSpace,
    TabularInputSpace,
    CombinedInputSpace,
)
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

    variable_domains = OrderedDict(variable_domains)
    full_input_space = CombinedInputSpace(variable_domains)

    # simplify the probability subject
    variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        var: domain.input_bounds for var, domain in variable_domains.items()
    }
    subj, variable_bounds = apply_symbolic_bounds(probability.subject, variable_bounds)

    # add batch dimensions to all bounds
    variable_bounds = {
        var: (lbs.unsqueeze(0), ubs.unsqueeze(0))
        for var, (lbs, ubs) in variable_bounds.items()
    }

    requires_bounds = collect_requires_bounds(subj)
    bounded_terms_substitution = {
        term: ExternalVariable(str(term)) for term in requires_bounds
    }
    subj_skeleton = subj.replace(bounded_terms_substitution)

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

    def probability_mass(
        var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        return prod(
            variable_distributions[var].cdf(var_ubs)
            - variable_distributions[var].cdf(var_lbs)
            for var, (var_lbs, var_ubs) in var_bounds.items()
        )

    branches = BranchStore(
        in_shape=full_input_space.input_shape,
        out_shape=(1,),  # a satisfaction score
        probability_mass=(1,),
    )
    intermediate_bounds = {
        str(term): eval_bounds(term, variable_bounds) for term in requires_bounds
    }
    subj_skeleton_sat_fn = subj_skeleton.satisfaction_function
    sat_lb, sat_ub = subj_skeleton_sat_fn.propagate_bounds(**intermediate_bounds)
    prob_mass = probability_mass(variable_bounds)
    in_lbs = {var: lbs for var, (lbs, _) in variable_bounds.items()}
    in_ubs = {var: ubs for var, (_, ubs) in variable_bounds.items()}
    branches.append(
        in_lbs=full_input_space.combine(**in_lbs),
        in_ubs=full_input_space.combine(**in_ubs),
        out_lbs=sat_lb,
        out_ubs=sat_ub,
        probability_mass=prob_mass,
    )

    def eval_probability_bounds() -> tuple[torch.Tensor, torch.Tensor]:
        # satisfaction lower bound > 0 => sufficient condition for satisfaction
        prob_lb = torch.sum(
            torch.where(branches.out_lbs > 0, branches.probability_mass, 0)
        )
        # satisfaction upper bound > 0 => necessary condition for satisfaction
        # the branches with satisfaction upper bound > 0 are those for which
        # satisfaction hasn't yet been disproven.
        prob_ub = torch.sum(
            torch.where(branches.out_ubs > 0, branches.probability_mass, 0)
        )
        return prob_lb, prob_ub

    best_lb, best_ub = eval_probability_bounds()
    yield (best_lb, best_ub)

    if len(networks) == 0:
        # No need to refine bounds in this case
        while True:
            yield (best_lb, best_ub)

    while True:
        # 1. select a batch of branches
        branches.sort(branches.probability_mass.flatten(), descending=True)
        selected_branches = branches.pop(batch_size)

        # 2. select dimensions to split
        if split_heuristic.upper() == "IBP":
            # TODO:
            splits = split_ibp(selected_branches, full_input_space)
        elif split_heuristic.lower() == "longest-edge":
            splits = split_longest_edge(selected_branches, full_input_space)
        else:
            raise ValueError(
                f"Unknown split heuristic: {split_heuristic}."
                f"Use either 'IBP' or 'longest-edge'."
            )

        # 3. split branches
        left_lbs, left_ubs = splits.left_branch
        right_lbs, right_ubs = splits.right_branch
        new_lbs = torch.vstack([left_lbs, right_lbs])
        new_ubs = torch.vstack([left_ubs, right_ubs])

        # 4. compute bounds
        variable_lbs = full_input_space.decompose(new_lbs)
        variable_ubs = full_input_space.decompose(new_ubs)
        variable_bounds = {
            var: (lbs, variable_ubs[var]) for var, lbs in variable_lbs.items()
        }
        intermediate_bounds = {
            str(term): eval_bounds(term, variable_bounds) for term in requires_bounds
        }
        sat_lbs, sat_ubs = subj_skeleton_sat_fn.propagate_bounds(**intermediate_bounds)
        prob_mass = probability_mass(variable_bounds)

        # 5. update branches
        branches.append(
            in_lbs=new_lbs,
            in_ubs=new_ubs,
            out_lbs=sat_lbs.unsqueeze(1),
            out_ubs=sat_ubs.unsqueeze(1),
            probability_mass=prob_mass,
        )

        # 6. report new lower/upper bound on probability
        best_lb, best_ub = eval_probability_bounds()
        yield (best_lb, best_ub)


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
        simple_terms = []
        remaining_terms = []
        # This AND may contain nested ANDs
        top_level_conjuncts = [formula]
        while len(top_level_conjuncts) > 0:
            conj = top_level_conjuncts.pop()
            for term in conj.operands:
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


@dataclass
class Split:
    """
    The two branches of a split. Each branch is represented by
    a lower and an upper bound.
    """

    left_branch: tuple[torch.Tensor, torch.Tensor]
    right_branch: tuple[torch.Tensor, torch.Tensor]

    @staticmethod
    def stack(splits: Sequence["Split"], dim=0) -> "Split":
        """
        Stacks (:func:`torch.stack`) the bounds tensors of
        several :class:`Split` instances.
        Produces a batched :class:`Split` representing several splits.

        :param splits: The :class:`Split` instances to stack.
        :param dim: See :code:`torch.stack`.
        :return: A batched :class:`Split`.
        """
        kwargs = {}
        for branch in ("left_branch", "right_branch"):
            lbs = []
            ubs = []
            for s in splits:
                lb, ub = getattr(s, branch)
                lbs.append(lb)
                ubs.append(ub)
            kwargs[branch] = (torch.stack(lbs, dim), torch.stack(ubs, dim))
        return Split(**kwargs)

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Split":
        """
        Applies a function :code:`fn` to the `mask` and `values` tensors of
        the :class:`TensorUpdate` instances underlying this :class:`Split`.
        :param fn: The function to apply to the underlying `mask` and `values` tensors.
        :return: A :class:`Split` with :code:`fn` applied to the `mask` and `values`
         tensors.
        """
        kwargs = {}
        for branch in ("left_branch", "right_branch"):
            lb, ub = getattr(self, branch)
            kwargs[branch] = (fn(lb), fn(ub))
        return Split(**kwargs)


def propose_splits(
    branches: BranchStore, combined_input_space: CombinedInputSpace
) -> tuple[Split, torch.Tensor]:
    """
    Creates all possible splits for a batch of branches.
    While continuous input attributes can be split arbitrarily often,
    categorical input attributes can only be split into the
    different values of the categorical attribute.
    Overall, the number of proposed splits is the number of dimensions
    of the :code:`combined_input_space`.

    :param branches: A batch of branches for which to determine splits.
    :param combined_input_space: The combined input space of all variables.
    :return: A batch of splits and a tensor of booleans indicating invalid splits.
     Invalid splits are, for example, splits that split on the value of a categorical
     variable that was already split.

     The shape of the bounds inside the :code:`Split`
     is `(num_splits, batch_dim, input_dims)`.
     The index of the first dimension corresponds to the dimension of the
     combined input space that is split.
     This means that :code:`propose_splits(...)[i]` splits the i-th dimension
     of the combined input space.
     The tensor indicating invalid splits has the shape (num_splits, batch_dim).
     The i,j-th entry of the tensor is :code:`True` if the i-th split is invalid
     for the batch entry j.
    """
    lbs_flat = branches.in_lbs
    ubs_flat = branches.in_ubs

    def continuous_split(dim_) -> tuple[Split, torch.Tensor]:
        midpoint = (ubs_flat[:, dim_] + lbs_flat[:, dim_]) / 2.0
        # two splits: set lower bound to midpoint
        # and set upper bound to midpoint
        left_ubs = ubs_flat.detach().clone()
        right_lbs = lbs_flat.detach().clone()
        left_ubs[:, dim_] = midpoint
        right_lbs[:, dim_] = midpoint
        split = Split(
            left_branch=(lbs_flat, left_ubs), right_branch=(right_lbs, ubs_flat)
        )
        # any split is valid, so return all false for the invalid splits tensor
        is_invalid_ = torch.zeros(
            lbs_flat.size(0), dtype=torch.bool, device=lbs_flat.device
        )
        return split, is_invalid_

    def categorical_split(dims_, val_i_) -> tuple[Split, torch.Tensor]:
        """
        :param dims_: The dimensions of the input space that together make up
         the categorical attribute.
        :param val_i_: The index of the value to split on.
        """
        # two splits: set to the value and exclude the value

        # Left branch: set the value
        # => assign lbs and ubs of all other values to 0.0
        #    and assign lb and ub for val_i to 1.0.
        left_lbs = lbs_flat.detach().clone()
        left_ubs = ubs_flat.detach().clone()
        left_lbs[:, dims_] = 0.0
        left_lbs[:, dims_[val_i_]] = 1.0
        left_ubs[:, dims_] = 0.0
        left_ubs[:, dims_[val_i_]] = 1.0

        # Right branch: exclude this value
        # => set lb and ub for this value to 0.0.
        #    No change to other values.
        right_lbs = lbs_flat.detach().clone()
        right_ubs = ubs_flat.detach().clone()
        right_lbs[:, dims_[val_i_]] = 0.0
        right_ubs[:, dims_[val_i_]] = 0.0
        # But: may not exclude all values.
        # If there is only one value remaining after excluding this value,
        # we need to set the remaining value.
        dims_mask = torch.zeros_like(lbs_flat).bool()
        dims_mask[:, dims] = True
        unset_values = (right_lbs != right_ubs) & dims_mask
        num_unset_values = unset_values.float().sum(dim=1)
        num_unset_values.unqueeze_(1)
        right_lbs = torch.where(num_unset_values == 1 & unset_values, 1.0, right_lbs)

        # If this value is already assigned (either set or excluded from assignment)
        # then we can't branch on this value any more.
        # Mark these splits invalid.
        this_value_already_set: torch.Tensor = (
            lbs_flat[:, dims_[val_i_]] == ubs_flat[:, dims_[val_i_]]
        )

        return (
            Split((left_lbs, left_ubs), (right_lbs, right_ubs)),
            this_value_already_set,
        )

    splits: list[Split | None] = [None] * combined_input_space.input_shape[0]
    is_invalid: list[torch.Tensor | None] = [None] * len(splits)
    for var, offset in zip(
        combined_input_space.variables, combined_input_space.offsets
    ):
        var_domain = combined_input_space.domain_of(var)
        if isinstance(var_domain, TensorInputSpace):
            for dim in range(combined_input_space.dimensions_of(var)):
                split, invalid = continuous_split(offset + dim)
                splits[offset + dim] = split
                is_invalid[offset + dim] = invalid
        elif isinstance(var_domain, TabularInputSpace):
            layout = var_domain.encoding_layout
            for attr_i in range(len(var_domain)):
                attr_name = var_domain.attribute_name(attr_i)
                match var_domain.attribute_type(attr_i):
                    case TabularInputSpace.AttributeType.CONTINUOUS:
                        dim = offset + layout[attr_name]
                        split, invalid = continuous_split(dim)
                        splits[dim] = split
                        is_invalid[dim] = invalid
                    case TabularInputSpace.AttributeType.CATEGORICAL:
                        dims = [offset + dim for dim in layout.values()]
                        for val_i, val in enumerate(
                            var_domain.attribute_values(attr_i)
                        ):
                            val_dim = offset + layout[attr_name][val]
                            split, invalid = categorical_split(dims, val_i)
                            splits[val_dim] = split
                            is_invalid[val_dim] = invalid
                    case _:
                        raise ValueError(
                            f"Attribute {attr_name} has unknown "
                            f"attribute type {var_domain.attribute_type(attr_i)}"
                        )
        else:
            raise ValueError(
                f"Unknown input space type {type(var_domain)}. "
                f"Supported types: TensorInputSpace and TabularInputSpace."
            )

    return Split.stack(splits), torch.stack(is_invalid)


def split_ibp(branches: BranchStore, combined_input_space: CombinedInputSpace) -> Split:
    # TODO
    #  - IBP: select dimension by improvement in satisfaction score bounds
    raise NotImplementedError()


def split_longest_edge(
    branches: BranchStore, combined_input_space: CombinedInputSpace
) -> Split:
    """
    Select input dimensions to split.
    :code:`split_longest_edge` selects the dimension (for each batch element)
    that has the largest distance between lower and upper bound (the longest edge).

    :param branches: The branches for which to determine the dimensions to split.
    :param combined_input_space: The combined input space of all variables.
    :return: A split to perform and the value of the split.
     The value of the split is the edge length of the split dimension.
     Dimensions of categorical attributes for tabular input spaces generally
     have edge length 1.
    """
    # 1st dim of tensors in splits corresponds to the dimension that is split
    splits, is_invalid = propose_splits(branches, combined_input_space)

    edge_len = branches.in_ubs - branches.in_lbs
    edge_len.masked_fill_(is_invalid.T, -torch.inf)
    split_dims = torch.argmax(edge_len, dim=1)
    split_dims = split_dims.reshape(1, -1, 1)

    def take_selected(t: torch.Tensor):
        return t.take_along_dim(split_dims, dim=0).squeeze(0)

    return splits.map(take_selected)
