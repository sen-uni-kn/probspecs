# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import typing
from collections import OrderedDict
from dataclasses import dataclass
from math import prod
from typing import Generator, Literal, Sequence, Callable, Final

import torch
from frozendict import frozendict
from torch import nn
from auto_LiRPA import BoundedModule

from .auto_lirpa_params import AutoLiRPAParams
from .branch_store import BranchStore
from ..trinary_logic import TrinaryLogic
from ..formula import (
    ExternalFunction,
    Function,
    Probability,
    ExternalVariable,
    Constant,
    Inequality,
    Formula,
    ElementAccess,
)
from ..distributions.probability_distribution import ProbabilityDistribution
from ..input_space import (
    InputSpace,
    TensorInputSpace,
    TabularInputSpace,
    CombinedInputSpace,
)
from .utils import construct_bounded_tensor
from ..utils.formula_utils import (
    collect_requires_bounds,
    fuse_compositions,
    make_explicit,
)

__all__ = [
    "SPLIT_HEURISTICS",
    "SPLIT_HEURISTICS_TYPE",
    "probability_bounds",
    "split_longest_edge",
    "split_random",
    "split_prob_balanced",
    "smart_branching",
]


SPLIT_HEURISTICS_TYPE: Final = Literal[
    "IBP",
    "CROWN",
    "longest-edge",
    "normalized-longest-edge",
    "random",
    "prob-balanced",
]
SPLIT_HEURISTICS: Final[tuple[SPLIT_HEURISTICS_TYPE, ...]] = typing.get_args(
    SPLIT_HEURISTICS_TYPE
)


def probability_bounds(
    probability: Probability,
    networks: dict[str, nn.Module],
    variable_domains: dict[str, InputSpace],
    variable_distributions: dict[str, ProbabilityDistribution],
    batch_size: int = 128,
    auto_lirpa_params: AutoLiRPAParams = AutoLiRPAParams(),
    split_heuristic: SPLIT_HEURISTICS_TYPE = "normalized-longest-edge",
    split_heuristic_params: dict[Literal["better_branch"], bool] = frozendict(),
    device: str | torch.device | None = None,
) -> Generator[tuple[float, float], None, None]:
    """
    Computes a sequence of refined bounds on a probability.
    With each yield of this generator, the lower and upper bounds that it
    produces improve, meaning that the lower bound increases while the upper
    bound decreases.

    To refine the previously computed bounds, :code:`probability_bounds` performs
    branch and bound with input splitting.

    Split heuristics:
     - :code:`"IBP"`: smart branching with IBP.
        Split the dimension that leads to the best IBP bounds.
        This heuristic has an additional parameter :code:`better_branch`
        that determines which branch that results from a split is selected
        to evaluate the split.
        You can specify this parameter using the :code:`split_heuristic_params`
        argument.
        By default (:code:`better_branch=True`), a split is evaluated using the
        bounds of the resulting branch with the better bounds.
        When :code:`better_branch=True`, instead the branch with the worse bounds
        is used.
        This is the strategy of BaBSR [BunelEtAl2020]_.
     - :code:`"CROWN"`: smart branching with CROWN.
        Split the dimension that leads to the best CROWN bounds.
        This split heuristic also has a :code:`better_branch? parameter.
        See :code:`IBP` for more details on this parameter.
     - :code:`"longest-edge"`: Split the dimension that has the largest diameter.
     - :code:`"normalized-longest-edge"`: Split the dimension that has the largest
        diameter, where the diameter of each dimension is normalized by the initial
        diameter of the dimension.
        This variant of longest edge splitting is recommended for input spaces whose
        dimensions have different scales.
     - :code:`"prob-balanced"`: Tries to balance the probability mass between branches.
     - :code:`"random"`: Randomly selects a dimension to split.

    .. [BunelEtAl2020] Rudy Bunel, Jingyue Lu, Ilker Turkaslan, Philip H. S. Torr,
        Pushmeet Kohli, M. Pawan Kumar: Branch and Bound for Piecewise Linear
        Neural Network Verification. J. Mach. Learn. Res. 21: 42:1-42:39 (2020)

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
    :param split_heuristic_params: Additional parameters of the split heuristic.
    :param device: The device to compute on.
     If None, the tensors remain on the device they already reside on.
    :return: A generator that yields improving lower and upper bounds.
    """
    if split_heuristic not in SPLIT_HEURISTICS:
        raise ValueError(
            f"Unknown split heuristic: {split_heuristic}."
            f"Use one from {', '.join(SPLIT_HEURISTICS)}."
        )

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
            split_heuristic_params,
            device,
        )
        p_condition_bounds = probability_bounds(
            p_condition,
            networks,
            variable_domains,
            variable_distributions,
            batch_size,
            auto_lirpa_params,
            split_heuristic,
            split_heuristic_params,
            device,
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
    subj, _ = make_explicit(probability.subject, **networks)
    subj, _ = fuse_compositions(subj)
    subj, variable_bounds = apply_symbolic_bounds(subj, variable_bounds)

    # add batch dimensions to all bounds
    variable_bounds = {
        var: (lbs.unsqueeze(0).to(device), ubs.unsqueeze(0).to(device))
        for var, (lbs, ubs) in variable_bounds.items()
    }

    requires_bounds = set(collect_requires_bounds(subj))
    bounded_terms_substitution = {
        term: ExternalVariable(f"?{term}?") for term in requires_bounds
    }
    subs_names = {term: var.name for term, var in bounded_terms_substitution.items()}
    subj_skeleton = subj.replace(bounded_terms_substitution)
    subj_skeleton_sat_fn = subj_skeleton.satisfaction_function

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
            external.get_function(**networks),
            variable_bounds[external.arg_names[0]][0],  # lower bound of arg
            auto_lirpa_params.bound_ops,
            device,
        )
        for external in requires_bounds
        if isinstance(external, ExternalFunction)
    }

    @torch.no_grad()
    def propagate_bounds(
        term_: Function,
        var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
        method: str = auto_lirpa_params.method,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(term_, ExternalVariable):
            return term_.propagate_bounds(**var_bounds)
        elif isinstance(term_, ExternalFunction):
            network = networks[term_.func_name]
            bounded_tensor = construct_bounded_tensor(*var_bounds[term_.arg_names[0]])
            return network.compute_bounds(x=(bounded_tensor,), method=method)
        else:
            raise NotImplementedError()

    def sat_bounds(
        var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
        method: str = auto_lirpa_params.method,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns lower and upper bounds on the satisfaction function
        (first two tensors) and a tensor of trinary truth values indicating
        whether :code:`subj_skeleton` is satisfied.

        The additional trinary logic values are useful since a satisfaction function
        value of 0.0 is inconclusive.
        """
        intermediate_bounds = {
            subs_names[term]: propagate_bounds(term, var_bounds, method)
            for term in requires_bounds
        }
        sat_lb_, sat_ub_ = subj_skeleton_sat_fn.propagate_bounds(**intermediate_bounds)
        is_sat = subj_skeleton.propagate_bounds(**intermediate_bounds)
        if not isinstance(is_sat, torch.Tensor):
            is_sat = torch.tensor([is_sat])
        sat_lb_ = sat_lb_.to(device=device)
        sat_ub_ = sat_ub_.to(device=device)
        is_sat = is_sat.to(device=device)
        return sat_lb_, sat_ub_, is_sat

    @torch.no_grad()
    def probability_mass(
        var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        probs = (
            variable_distributions[var].probability(bounds)
            for var, bounds in var_bounds.items()
        )
        return prod(probs).reshape(-1, 1)

    branches = BranchStore(
        in_shape=full_input_space.input_shape,
        out_shape=(1,),  # a satisfaction score
        is_sat=(1,),  # trinary logic values for actual satisfaction
        probability_mass=(1,),
        device=device,
    )
    sat_lb, sat_ub, is_sat = sat_bounds(variable_bounds)
    prob_mass = probability_mass(variable_bounds)
    in_lbs = {var: lbs for var, (lbs, _) in variable_bounds.items()}
    in_ubs = {var: ubs for var, (_, ubs) in variable_bounds.items()}
    branches.append(
        in_lbs=full_input_space.combine(**in_lbs),
        in_ubs=full_input_space.combine(**in_ubs),
        out_lbs=sat_lb,
        out_ubs=sat_ub,
        is_sat=is_sat,
        probability_mass=prob_mass,
    )

    # prob_lb = probability mass in all regions where sat_lb > 0, which is a
    # sufficient condition for satisfaction.
    prob_lb = torch.tensor(0.0, dtype=prob_mass.dtype, device=device)
    # prob_ub = 1 - probability mass in all regions where sat_ub < 0, which is a
    # sufficient condition for violation.
    prob_ub = torch.sum(branches.probability_mass)  # total probability may be < 1.0

    # Some split heuristics resolve ties using randomness to avoid that the
    # first dimensions are split more frequently than later dimensions.
    split_heuristic_rng = torch.Generator(device=device)
    split_heuristic_rng.manual_seed(513733474145742)
    while True:
        # 0. Remove branches where subj is certainly satisfied or certainly violated.
        # For certainly violated branches, add their probability mass to prob_lb and
        # remove the branches.
        certainly_sat_mask = TrinaryLogic.is_true(branches.is_sat)
        prob_lb += torch.sum(certainly_sat_mask * branches.probability_mass)
        branches.drop(certainly_sat_mask)
        # Remove certainly violated branches.
        certainly_viol_mask = TrinaryLogic.is_false(branches.is_sat)
        prob_ub -= torch.sum(certainly_viol_mask * branches.probability_mass)
        branches.drop(certainly_viol_mask)

        print(f"Probability Bounds ({subj}): lb={prob_lb}, ub={prob_ub}")
        yield (prob_lb, prob_ub)

        # this is primarily for the case when apply_symbolic_bounds
        # removed all terms from subj
        # (for example, subj only contains bounds on variables)
        if torch.isclose(prob_lb, prob_ub, atol=0.0, rtol=1e-5):
            break

        if len(branches) == 0:
            raise RuntimeError(
                f"No branches left, but bounds did not converge: {prob_lb} < {prob_ub}."
            )

        # 1. Select a batch of branches
        branches.sort(branches.probability_mass.flatten(), descending=True)
        selected_branches = branches.pop(batch_size)

        # 2. Select dimensions to split
        if split_heuristic.casefold() in ("ibp", "crown"):
            splits = smart_branching(
                selected_branches,
                full_input_space,
                lambda *args: sat_bounds(*args)[:2],
                method=split_heuristic.casefold().upper(),
                rng=split_heuristic_rng,
                **split_heuristic_params,
            )
        elif split_heuristic.casefold().endswith("longest-edge"):
            splits = split_longest_edge(
                selected_branches,
                full_input_space,
                normalize=split_heuristic.casefold().startswith("normalized"),
            )
        elif split_heuristic.casefold() == "prob-balanced":
            splits = split_prob_balanced(
                selected_branches, full_input_space, probability_mass
            )
        elif split_heuristic.casefold() == "random":
            splits = split_random(
                selected_branches, full_input_space, split_heuristic_rng
            )
        else:
            raise NotImplementedError()

        # 3. Split branches
        left_lbs, left_ubs = splits.left_branch
        right_lbs, right_ubs = splits.right_branch
        new_lbs = torch.vstack([left_lbs, right_lbs])
        new_ubs = torch.vstack([left_ubs, right_ubs])

        # 4. Compute bounds
        in_lbs = full_input_space.decompose(new_lbs)
        in_ubs = full_input_space.decompose(new_ubs)
        variable_bounds = {var: (in_lbs[var], in_ubs[var]) for var in variable_bounds}
        sat_lbs, sat_ubs, is_sat = sat_bounds(variable_bounds)
        prob_mass = probability_mass(variable_bounds)

        # 5. Update branches
        branches.append(
            in_lbs=new_lbs,
            in_ubs=new_ubs,
            out_lbs=sat_lbs.unsqueeze(1),
            out_ubs=sat_ubs.unsqueeze(1),
            is_sat=is_sat.unsqueeze(1),
            probability_mass=prob_mass,
        )

    while True:
        yield (prob_lb, prob_ub)


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
        """
        Simple terms: x >= c, x <= c, c >= x, c <= x,
        x[...] >= c, x[...] <= c, c >= x[...], c <= x[...]
        where c is a Constant and x is an ExternalVariable.
        """
        if isinstance(term_, Inequality) and term_.op in (
            Inequality.Operator.LESS_EQUAL,
            Inequality.Operator.GREATER_EQUAL,
        ):
            if isinstance(term_.lhs, Constant):
                const_ = term_.lhs
                other_ = term_.rhs
            elif isinstance(term_.rhs, Constant):
                const_ = term_.rhs
                other_ = term_.lhs
            else:
                const_ = other_ = None

            if const_ is not None:
                return isinstance(other_, ExternalVariable) or (
                    isinstance(other_, ElementAccess)
                    and isinstance(other_.source, ExternalVariable)
                )
        return False

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
                    top_level_conjuncts.append(term)
                else:
                    remaining_terms.append(term)
    elif is_simple(formula):
        simple_terms = [formula]
        remaining_terms = []
    else:
        return formula, variable_bounds

    new_bounds = dict(variable_bounds)
    for ineq in simple_terms:
        const = ineq.lhs if isinstance(ineq.lhs, Constant) else ineq.rhs
        other = ineq.rhs if isinstance(ineq.lhs, Constant) else ineq.lhs

        if isinstance(other, ExternalVariable):
            var = other
            lb, ub = new_bounds[var.name]
            if ineq.op is Inequality.Operator.LESS_EQUAL:
                ub = torch.clamp(ub, max=const.val)
            else:  # var >= const
                lb = torch.clamp(lb, min=const.val)
            new_bounds[var.name] = (lb, ub)
        elif isinstance(other, ElementAccess):
            var = other.source
            lb, ub = new_bounds[var.name]
            # deal with missing batch dimensions in bounds
            item = other.target_item[1:]
            lb_item = lb[item]
            ub_item = ub[item]
            if ineq.op is Inequality.Operator.LESS_EQUAL:
                ub_item = torch.clamp(ub_item, max=const.val)
            else:  # var >= const
                lb_item = torch.clamp(lb_item, min=const.val)
            lb[item] = lb_item
            ub[item] = ub_item
            new_bounds[var.name] = (lb, ub)
        else:
            raise NotImplementedError()

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

    def select(self, selected_splits: torch.Tensor) -> "Split":
        """
        Select a split for each batch element.

        :param selected_splits: A 1d tensor containing the index of one
         split to select for each batch element.
        :return: A :class:`Split` object containing the selected splits.
        """
        selected_splits = selected_splits.reshape(1, -1, 1)

        def take_selected(t: torch.Tensor):
            return t.take_along_dim(selected_splits, dim=0).squeeze(0)

        return self.map(take_selected)


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
        # The split is invalid if the bounds are already tight.
        is_invalid_ = ubs_flat[:, dim_] == lbs_flat[:, dim_]
        return split, is_invalid_

    def integer_split(dim_) -> tuple[Split, torch.Tensor]:
        midpoint = (ubs_flat[:, dim_] + lbs_flat[:, dim_]) / 2.0
        # two splits: set lower bound to midpoint
        # and set upper bound to midpoint
        left_ubs = ubs_flat.detach().clone()
        right_lbs = lbs_flat.detach().clone()
        left_ubs[:, dim_] = torch.floor(midpoint)
        right_lbs[:, dim_] = torch.ceil(midpoint)
        split = Split(
            left_branch=(lbs_flat, left_ubs), right_branch=(right_lbs, ubs_flat)
        )
        # The split is invalid if the bounds are already tight.
        is_invalid_ = ubs_flat[:, dim_] == lbs_flat[:, dim_]
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
        num_unset_values.unsqueeze_(1)
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
                    case TabularInputSpace.AttributeType.INTEGER:
                        dim = offset + layout[attr_name]
                        split, invalid = integer_split(dim)
                        splits[dim] = split
                        is_invalid[dim] = invalid
                    case TabularInputSpace.AttributeType.CATEGORICAL:
                        dims = [offset + dim for dim in layout[attr_name].values()]
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


def _get_branch_bounds(splits, combined_input_space):
    left_lbs, left_ubs = splits.left_branch
    right_lbs, right_ubs = splits.right_branch
    num_splits, batch_size, input_size = left_lbs.shape
    left_lbs = left_lbs.reshape(-1, input_size)
    left_ubs = left_ubs.reshape_as(left_lbs)
    right_lbs = right_lbs.reshape_as(left_lbs)
    right_ubs = right_ubs.reshape_as(left_lbs)
    left_lbs = combined_input_space.decompose(left_lbs)
    left_ubs = combined_input_space.decompose(left_ubs)
    right_lbs = combined_input_space.decompose(right_lbs)
    right_ubs = combined_input_space.decompose(right_ubs)
    left_bounds = {var: (left_lbs[var], left_ubs[var]) for var in left_lbs}
    right_bounds = {var: (right_lbs[var], right_ubs[var]) for var in left_lbs}
    return left_bounds, right_bounds, num_splits, batch_size


@torch.no_grad()
def smart_branching(
    branches: BranchStore,
    combined_input_space: CombinedInputSpace,
    compute_sat_bounds: Callable[
        [dict[str, tuple[torch.Tensor, torch.Tensor]], str],
        tuple[torch.Tensor, torch.Tensor],
    ],
    method: Literal["IBP", "CROWN"],
    better_branch: bool = True,
    rng: torch.Generator = torch.Generator(),
) -> Split:
    """
    Select input dimensions to split.
    :code:`smart_branching` selects the dimension (for each batch element)
    that leads to the largest improvement in the estimated lower or upper bound
    of the satisfaction function when split.
    The improvements in lower and upper bounds are estimated using the
    bounding algorithm :code:`method`.
    You should select a cheap bounding algorithm, such as IBP for this task.
    This split strategy is an adaption of BaBSR from [BunelEtAl2020]_.

    Each split is evaluated, either by selecting the branch resulting from the split
    that has the better or worse bounds.
    Which is selected is determined by the :code:`better_branch` argument.
    By default, the better bounds are used, but using the worse bounds may lead to
    a more balanced branching tree.

    .. [BunelEtAl2020] Rudy Bunel, Jingyue Lu, Ilker Turkaslan, Philip H. S. Torr,
        Pushmeet Kohli, M. Pawan Kumar: Branch and Bound for Piecewise Linear
        Neural Network Verification. J. Mach. Learn. Res. 21: 42:1-42:39 (2020)

    :param branches: The branches for which to determine the dimensions to split.
    :param combined_input_space: The combined input space of all variables.
    :param compute_sat_bounds: Computes the bounds of the satisfaction function
     for a set of variable bounds (first argument).
     The second argument specifies the method to use for computing the bounds.
    :param method: The method to use for computing bounds.
    :param better_branch: Whether to select the branch with the better
     bounds (:code:`True`) or the branch with the worse bounds (:code:`False`)
     for evaluating a split.
    :param rng: A random number generator for breaking ties among splits
     that lead equally large improvements.
    :return: The splits to perform.
    """
    splits, is_invalid = propose_splits(branches, combined_input_space)
    left_bounds, right_bounds, num_splits, batch_size = _get_branch_bounds(
        splits, combined_input_space
    )

    # We already use very large batch sizes here since left_lbs contains
    # num input dims * original batch size many elements.
    # To balance memory requirements better, it makes sense to perform ibp twice here
    # instead of one call with an extremely large batch size.
    left_sat_lbs, left_sat_ubs = compute_sat_bounds(left_bounds, method)
    right_sat_lbs, right_sat_ubs = compute_sat_bounds(right_bounds, method)

    # recreate (split dims, batch) shape (individual sat bounds are scalars)
    left_sat_lbs = left_sat_lbs.reshape(num_splits, batch_size)
    left_sat_ubs = left_sat_ubs.reshape(num_splits, batch_size)
    right_sat_lbs = right_sat_lbs.reshape(num_splits, batch_size)
    right_sat_ubs = right_sat_ubs.reshape(num_splits, batch_size)

    # give invalid splits maximally bad sat scores so that they aren't selected.
    left_sat_lbs.masked_fill_(is_invalid, -torch.inf)
    right_sat_lbs.masked_fill_(is_invalid, -torch.inf)
    left_sat_ubs.masked_fill_(is_invalid, torch.inf)
    right_sat_ubs.masked_fill_(is_invalid, torch.inf)

    # Split selection strategy:
    # Select the split which makes the lower/upper bound on the branch
    # with the smaller/larger lower/upper bound largest/smallest.
    # Branches which make the lower bound positive or the upper bound negative
    # are the best splits, as they allow for pruning the branch.
    #
    # Adapted from: BaBSR (Rudy Bunel, Jingyue Lu, Ilker Turkaslan, Philip H. S. Torr,
    # Pushmeet Kohli, M. Pawan Kumar: Branch and Bound for Piecewise Linear
    # Neural Network Verification. J. Mach. Learn. Res. 21: 42:1-42:39 (2020))
    if better_branch:
        score_sat_lbs = torch.maximum(left_sat_lbs, right_sat_lbs)
        score_sat_ubs = torch.minimum(left_sat_ubs, right_sat_ubs)
    else:
        score_sat_lbs = torch.minimum(left_sat_lbs, right_sat_lbs)
        score_sat_ubs = torch.maximum(left_sat_ubs, right_sat_ubs)
    select_score = torch.maximum(score_sat_lbs, -score_sat_ubs)
    # At times splitting bounds that actually influence the output can lead
    # to slightly worse bounds than splitting an input that does not influence
    # the output due to floating point error.
    # To avoid this, we round the scores.
    select_score = torch.round(select_score, decimals=4)
    # resolve ties in select_scores randomly
    permute = torch.randperm(num_splits, generator=rng, device=rng.device)
    select_score = select_score[permute, :]
    split_dims = torch.argmax(select_score, dim=0)
    split_dims = permute[split_dims]
    return splits.select(split_dims)


def split_longest_edge(
    branches: BranchStore,
    combined_input_space: CombinedInputSpace,
    normalize: bool = False,
) -> Split:
    """
    Select input dimensions to split.
    :code:`split_longest_edge` selects the dimension (for each batch element)
    that has the largest distance between lower and upper bound (the longest edge).

    Optionally allows to normalize the edge lengths to the initial edge lengths
    as specified in :code:`combined_input_space`.

    :param branches: The branches for which to determine the dimensions to split.
    :param combined_input_space: The combined input space of all variables.
    :param normalize: Whether to apply edge length normalization.
    :return: The splits to perform.
    """
    # 1st dim of tensors in splits corresponds to the dimension that is split
    splits, is_invalid = propose_splits(branches, combined_input_space)

    edge_len = branches.in_ubs - branches.in_lbs
    if normalize:
        initial_lb, initial_ub = combined_input_space.input_bounds
        initial_lb, initial_ub = initial_lb.to(edge_len), initial_ub.to(edge_len)
        edge_len = edge_len / (initial_ub - initial_lb)
    edge_len.masked_fill_(is_invalid.T, -torch.inf)
    split_dims = torch.argmax(edge_len, dim=1)
    return splits.select(split_dims)


def split_prob_balanced(
    branches: BranchStore,
    combined_input_space: CombinedInputSpace,
    prob_mass: Callable[[dict[str, tuple[torch.Tensor, torch.Tensor]]], torch.Tensor],
) -> Split:
    """
    Select input dimensions to split.
    :code:`split_prob_balanced` selects to split the dimension which leads to the
    best balance in probability mass among the branches resulting from the split.
    Concretely, it selects the split minimizing the absolute difference in probability
    mass between the branches resulting from a split.

    :param branches: The branches for which to determine the dimensions to split.
    :param combined_input_space: The combined input space of all variables.
    :param prob_mass: Computes the probability mass within a region defined by
     bounds on the input variables.
    :return: The splits to perform.
    """
    splits, is_invalid = propose_splits(branches, combined_input_space)
    left_bounds, right_bounds, num_splits, batch_size = _get_branch_bounds(
        splits, combined_input_space
    )

    left_prob_mass = prob_mass(left_bounds)
    right_prob_mass = prob_mass(right_bounds)
    left_prob_mass = left_prob_mass.reshape(num_splits, batch_size)
    right_prob_mass = right_prob_mass.reshape(num_splits, batch_size)

    abs_difference = torch.abs(left_prob_mass - right_prob_mass)
    # give invalid branches maximally bad values, so that they aren't selected.
    abs_difference.masked_fill_(is_invalid, -torch.inf)
    split_dims = torch.argmin(abs_difference, dim=0)
    return splits.select(split_dims)


def split_random(
    branches: BranchStore,
    combined_input_space: CombinedInputSpace,
    rng: torch.Generator = torch.Generator(),
) -> Split:
    """
    Randomly input dimensions to split.

    :param branches: The branches for which to determine the dimensions to split.
    :param combined_input_space: The combined input space of all variables.
    :param rng: A random number generator for generating the splits.
    :return: The splits to perform.
    """
    splits, is_invalid = propose_splits(branches, combined_input_space)

    left_lbs, _ = splits.left_branch
    num_splits, batch_size, _ = left_lbs.shape

    branch_scores = torch.rand((num_splits, batch_size), generator=rng)
    branch_scores.masked_fill_(is_invalid, -torch.inf)
    split_dims = torch.argmax(branch_scores, dim=0)
    return splits.select(split_dims)
