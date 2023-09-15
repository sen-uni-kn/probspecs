# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Generator, Literal

from frozendict import frozendict
import torch
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


def refine_bounds(
    network: nn.Module,
    input_bounds: tuple[torch.Tensor, torch.Tensor],
    batch_size: int = 128,
    auto_lirpa_method: str = "alpha-CROWN",
    auto_lirpa_bound_ops=frozendict(
        {"optimize_bound_args": frozendict({"iteration": 20, "lr_alpha": 0.1})}
    ),
    split_heuristic: Literal["IBP", "longest-edge"] = "IBP",
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Computes a sequence of refined bounds for the output of :code:`network`.
    With each yield of this generator, the lower and upper bounds that it
    produces improve, meaning that the lower bound increases while the upper
    bound decreases.

    To refine the previously computed bounds, :code:`refine_bounds` performs
    branch and bound with input splitting.

    :param network: The network for which to compute bounds.
    :param input_bounds: A lower and an upper bound on the network input.
     The bounds may not have batch dimensions.
    :param batch_size: The number of branches to consider at a time.
    :param auto_lirpa_method: The :code:`auto_LiRPA` bound propagation method
     to use for computing bounds.
     More details in the :func:`auto_LiRPA.BoundedModule.compute_bounds` documentation.
    :param auto_lirpa_bound_ops: :code:`auto_LiRPA` bound propagation options.
     More details in the :func:`auto_LiRPA.BoundedModule` documentation.
    :param split_heuristic: Which heuristic to use for selecting dimensions to split.
    :return: A generator that yields improving lower and upper bounds.
    """
    initial_in_lb, initial_in_ub = input_bounds
    initial_in_lb = initial_in_lb.unsqueeze(0)
    initial_in_ub = initial_in_ub.unsqueeze(0)
    network = BoundedModule(network, initial_in_lb, auto_lirpa_bound_ops)
    bounded_tensor = construct_bounded_tensor(initial_in_lb, initial_in_ub)

    best_lb, best_ub = network.compute_bounds(
        x=(bounded_tensor,), method=auto_lirpa_method
    )
    yield (best_lb, best_ub)

    branch_in_bounds = torch.stack([initial_in_lb, initial_in_ub], dim=1)
    branch_out_bounds = torch.stack([best_lb, best_ub], dim=1)

    def score_branches():
        """
        A score how close the lb/ub in branch_bounds is to best_lb/best_ub.
        Branches with lower scores are selected for branching.
        """
        output_dims = tuple(range(2, branch_out_bounds.ndim))
        return torch.minimum(
            torch.amin(
                abs(best_lb - branch_out_bounds.index_select(1, torch.tensor(0))),
                dim=output_dims,
            ),
            torch.amin(
                abs(best_ub - branch_out_bounds.index_select(1, torch.tensor(1))),
                dim=output_dims,
            ),
        ).squeeze(1)

    while True:
        # 1. select a batch of branches
        branch_scores = score_branches()
        branch_scores_sorted = torch.sort(branch_scores)
        # each branch is later split into two branches
        # => select batch_size/2 branches for splitting
        this_batch_size = min(batch_size // 2, branch_scores.shape[0] - 1)
        selected_mask = torch.le(
            branch_scores, branch_scores_sorted[0][this_batch_size]
        )
        # since several branches may have the same score,
        # selected_mask may select too many branches
        for i in reversed(range(this_batch_size)):
            if sum(selected_mask) <= batch_size // 2:
                break
            selected_mask[branch_scores_sorted[1][i]] = False

        in_bounds_flat = branch_in_bounds.flatten(start_dim=2)
        in_bounds_selected_flat = in_bounds_flat[selected_mask, :, :]

        # 2. select dimensions to split
        if split_heuristic.upper() == "IBP":
            split_dims = split_ibp(
                network,
                in_bounds_selected_flat,
                branch_in_bounds.shape[2:],
                best_lb,
                best_ub,
            )
        elif split_heuristic.lower() == "longest-edge":
            split_dims = split_longest_edge(in_bounds_selected_flat)
        else:
            raise ValueError(
                f"Unknown split heuristic: {split_heuristic}."
                f"Use either 'IBP' or 'longest-edge'."
            )

        # 3. split branches
        split_dim_bounds = in_bounds_selected_flat.index_select(-1, split_dims)
        midpoints = (split_dim_bounds[:, 1, :] + split_dim_bounds[:, 0, :]) / 2
        lower_parts = in_bounds_selected_flat.detach().clone()
        upper_parts = in_bounds_selected_flat.detach().clone()
        lower_parts[:, 1, split_dims] = midpoints  # set upper bound to midpoints
        upper_parts[:, 0, split_dims] = midpoints  # set lower bound to midpoints
        split_in_bounds = torch.vstack([lower_parts, upper_parts])
        split_in_lbs, split_in_ubs = split_in_bounds[:, 0, :], split_in_bounds[:, 1, :]
        split_in_lbs = split_in_lbs.reshape(-1, *branch_in_bounds.shape[2:])
        split_in_ubs = split_in_ubs.reshape(-1, *branch_in_bounds.shape[2:])

        # 4. compute bounds
        bounded_tensor = construct_bounded_tensor(split_in_lbs, split_in_ubs)
        new_lbs, new_ubs = network.compute_bounds(
            x=(bounded_tensor,), method=auto_lirpa_method
        )

        # 5. update branches
        in_bounds_not_selected = in_bounds_flat[~selected_mask, :, :]
        in_bounds_not_selected = in_bounds_not_selected.reshape(
            -1, 2, *branch_in_bounds.shape[2:]
        )
        out_bounds_not_selected = branch_out_bounds[~selected_mask, :]

        split_in_bounds = split_in_bounds.reshape(-1, 2, *branch_in_bounds.shape[2:])
        branch_in_bounds = torch.vstack([in_bounds_not_selected, split_in_bounds])
        branch_out_bounds = torch.vstack(
            [out_bounds_not_selected, torch.stack([new_lbs, new_ubs]).unsqueeze(0)]
        )

        # 6. update best upper/lower bound
        best_lb = torch.amax(branch_out_bounds.index_select(1, 0), dim=0)
        best_ub = torch.amax(branch_out_bounds.index_select(1, 1), dim=0)
        yield (best_lb, best_ub)


def split_ibp(
    network: BoundedModule,
    in_bounds: torch.Tensor,
    input_shape: tuple[int, ...],
    curr_best_out_lb: torch.Tensor,
    curr_best_out_ub: torch.Tensor,
) -> torch.Tensor:
    """
    Select dimensions to split using Interval Bound Propagation (IBP).
    The selected dimensions are those that lead to the largest lower bounds
    or smallest upper bounds.

    :param network: The network for which bounds are computed.
    :param in_bounds: A tensor of lower and upper bounds on the input.
     The shape is :code:`(N, 2, M)` where :code:`N` is the batch size
     and :code:`M` is the number of dimensions of the input.
     You need to flatten the input before passing it to this function.
    :param input_shape: The actual shape of the input (unflattened).
    :param curr_best_out_lb: The currently best output lower bound.
    :param curr_best_out_ub: The currently best output upper bound.
    :return: A tensor of dimensions to split at their mid-point.
    """
    # split each dimension in half separately
    split = []
    for i in range(in_bounds.shape[-1]):
        midpoint = (in_bounds[:, 1, i] + in_bounds[:, 0, i]) / 2
        lower_part = in_bounds.detach().clone()
        upper_part = in_bounds.detach().clone()
        lower_part[:, 1, i] = midpoint  # set upper bound to midpoint
        upper_part[:, 0, i] = midpoint  # set lower bound to midpoint
        split.append(torch.stack([lower_part, upper_part]))
    split_in_bounds = torch.stack(split)  # shape: ndim, 2, N, 2, M
    # merge batch dim and splits dim
    split_batch = split_in_bounds.reshape(
        split_in_bounds[0] * 2 * split_in_bounds[2], 2, -1
    )
    in_lb_actual_shape = split_batch[:, 0, :].reshape(-1, *input_shape)
    in_ub_actual_shape = split_batch[:, 1, :].reshape(-1, *input_shape)
    bounded_tensor = construct_bounded_tensor(in_lb_actual_shape, in_ub_actual_shape)
    out_lbs, out_ubs = network.compute_bounds(x=(bounded_tensor,), method="IBP")
    # recreate split dims and batch dim
    out_lbs = out_lbs.reshape(
        split_in_bounds[0], 2, split_in_bounds[1], *out_lbs.shape[1:]
    )
    out_ubs = out_ubs.reshape(
        split_in_bounds[0], 2, split_in_bounds[1], *out_ubs.shape[1:]
    )
    smaller_out_lb = torch.amin(out_lbs, dim=1)
    larger_out_ub = torch.amax(out_ubs, dim=1)
    out_lb_improve = smaller_out_lb - curr_best_out_lb
    out_ub_improve = curr_best_out_ub - larger_out_ub
    larger_improve = torch.maximum(out_lb_improve, out_ub_improve)
    split_dims = torch.argmax(larger_improve, dim=0)
    return split_dims


def split_longest_edge(in_bounds: torch.Tensor) -> torch.Tensor:
    """
    Select dimensions to split.
    :code:`split_longest_edge` selects the dimension (for each batch element)
    that has the largest distance between lower and upper bound (the longest edge).

    :param in_bounds: A tensor of lower and upper bounds on the input.
     The shape is :code:`(N, 2, M)` where :code:`N` is the batch size
     and :code:`M` is the number of dimensions of the input.
     You need to flatten the input before passing it to this function.
    :return: A tensor of dimensions to split at their mid-point.
    """
    edge_len = in_bounds[:, 1, :] - in_bounds[:, 0, :]
    return torch.argmax(edge_len, dim=1)


def construct_bounded_tensor(in_lb: torch.Tensor, in_ub: torch.Tensor) -> BoundedTensor:
    input_domain = PerturbationLpNorm(x_L=in_lb, x_U=in_ub)
    midpoint = (in_ub + in_lb) / 2
    return BoundedTensor(midpoint, ptb=input_domain)
