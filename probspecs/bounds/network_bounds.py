# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Generator, Literal

import torch
from torch import nn
from auto_LiRPA import BoundedModule

from .auto_lirpa_params import AutoLiRPAParams
from .branch_store import BranchStore
from .utils import construct_bounded_tensor

__all__ = ["network_bounds", "split_ibp", "split_longest_edge"]


def network_bounds(
    network: nn.Module,
    input_bounds: tuple[torch.Tensor, torch.Tensor],
    batch_size: int = 128,
    auto_lirpa_params: AutoLiRPAParams = AutoLiRPAParams(),
    split_heuristic: Literal["IBP", "longest-edge"] = "IBP",
    device: str | torch.device | None = None,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Computes a sequence of refined bounds for the output of :code:`network`.
    With each yield of this generator, the lower and upper bounds that it
    produces improve, meaning that the lower bound increases while the upper
    bound decreases.

    To refine the previously computed bounds, :code:`network_bounds` performs
    branch and bound with input splitting.

    :param network: The network for which to compute bounds.
    :param input_bounds: A lower and an upper bound on the network input.
     The bounds may not have batch dimensions.
    :param batch_size: The number of branches to consider at a time.
    :param auto_lirpa_params: Parameters for running auto_LiRPA.
    :param split_heuristic: Which heuristic to use for selecting dimensions to split.
    :param device: The device to compute on.
     If None, the tensors remain on the device they already reside on.
    :return: A generator that yields improving lower and upper bounds.
    """
    initial_in_lb, initial_in_ub = input_bounds
    initial_in_lb = initial_in_lb.unsqueeze(0).to(device=device)
    initial_in_ub = initial_in_ub.unsqueeze(0).to(device=device)
    network = BoundedModule(network, initial_in_lb, auto_lirpa_params.bound_ops, device)
    bounded_tensor = construct_bounded_tensor(initial_in_lb, initial_in_ub)

    best_lb, best_ub = network.compute_bounds(
        x=(bounded_tensor,), method=auto_lirpa_params.method
    )
    yield (best_lb, best_ub)

    branches = BranchStore(initial_in_lb.shape[1:], best_lb.shape[1:], device)
    branches.append(
        in_lbs=initial_in_lb, in_ubs=initial_in_ub, out_lbs=best_lb, out_ubs=best_ub
    )

    def score_branches():
        """
        A score how close the lb/ub in branch_bounds is to best_lb/best_ub.
        Branches with lower scores are selected for branching.
        """
        output_dims = tuple(range(1, len(branches.output_shape) + 1))
        return torch.minimum(
            torch.amin(abs(best_lb - branches.out_lbs), dim=output_dims),
            torch.amin(abs(best_ub - branches.out_ubs), dim=output_dims),
        )

    while True:
        # 1. select a batch of branches
        branch_scores = score_branches()
        branches.sort(branch_scores, descending=False)
        selected_branches = branches.pop(batch_size)

        # 2. select dimensions to split
        if split_heuristic.upper() == "IBP":
            split_dims = split_ibp(
                network,
                selected_branches,
                best_lb,
                best_ub,
            )
        elif split_heuristic.lower() == "longest-edge":
            split_dims = split_longest_edge(selected_branches)
        else:
            raise ValueError(
                f"Unknown split heuristic: {split_heuristic}."
                f"Use either 'IBP' or 'longest-edge'."
            )

        # 3. split branches
        in_lbs_flat = selected_branches.in_lbs.flatten(1)
        in_ubs_flat = selected_branches.in_ubs.flatten(1)
        split_dim_lbs = in_lbs_flat.index_select(-1, split_dims)
        split_dim_ubs = in_ubs_flat.index_select(-1, split_dims)
        midpoints = (split_dim_ubs + split_dim_lbs) / 2.0
        # split into: lower part = [lb, mid] and upper part = [mid, ub]
        lower_part_ubs = in_ubs_flat.detach().clone()
        lower_part_ubs[:, split_dims] = midpoints
        upper_part_lbs = in_lbs_flat.detach().clone()
        upper_part_lbs[:, split_dims] = midpoints
        split_in_lbs = torch.vstack([in_lbs_flat, upper_part_lbs])
        split_in_ubs = torch.vstack([lower_part_ubs, in_ubs_flat])
        split_in_lbs = split_in_lbs.reshape(-1, *branches.input_shape)
        split_in_ubs = split_in_ubs.reshape(-1, *branches.input_shape)

        # 4. compute bounds
        bounded_tensor = construct_bounded_tensor(split_in_lbs, split_in_ubs)
        new_lbs, new_ubs = network.compute_bounds(
            x=(bounded_tensor,), method=auto_lirpa_params.method
        )

        # 5. update branches
        branches.append(
            in_lbs=split_in_lbs, in_ubs=split_in_ubs, out_lbs=new_lbs, out_ubs=new_ubs
        )

        # 6. update best upper/lower bound
        best_lb = torch.amin(branches.out_lbs, dim=0)
        best_ub = torch.amax(branches.out_ubs, dim=0)
        yield (best_lb, best_ub)


def split_ibp(
    network: BoundedModule,
    branches: BranchStore,
    curr_best_out_lb: torch.Tensor,
    curr_best_out_ub: torch.Tensor,
) -> torch.Tensor:
    """
    Select dimensions to split using Interval Bound Propagation (IBP).
    The selected dimensions are those that lead to the largest lower bounds
    or smallest upper bounds.

    :param network: The network for which bounds are computed.
    :param branches: The branches for which to determine the dimensions to split.
    :param curr_best_out_lb: The currently best output lower bound.
    :param curr_best_out_ub: The currently best output upper bound.
    :return: A tensor of dimensions to split at their mid-point.
     The dimensions are for a flattened input.
    """
    in_bounds = torch.stack([branches.in_lbs, branches.in_ubs], dim=1).flatten(2)

    # create the splits
    split = []
    for i in range(in_bounds.shape[-1]):
        midpoint = (in_bounds[:, 1, i] + in_bounds[:, 0, i]) / 2.0
        lower_part = in_bounds.detach().clone()
        upper_part = in_bounds.detach().clone()
        lower_part[:, 1, i] = midpoint  # set upper bound to midpoint
        upper_part[:, 0, i] = midpoint  # set lower bound to midpoint
        split.append(torch.stack([lower_part, upper_part]))
    split_in_bounds = torch.stack(split)  # shape: ndim, 2, N, 2, M
    # merge batch dim and splits dim
    split_batch = split_in_bounds.reshape(
        split_in_bounds.shape[0] * 2 * split_in_bounds.shape[2], 2, -1
    )

    # compute IBP bounds
    in_lb_actual_shape = split_batch[:, 0, :].reshape(-1, *branches.input_shape)
    in_ub_actual_shape = split_batch[:, 1, :].reshape(-1, *branches.input_shape)
    bounded_tensor = construct_bounded_tensor(in_lb_actual_shape, in_ub_actual_shape)
    out_lbs, out_ubs = network.compute_bounds(x=(bounded_tensor,), method="IBP")

    # recreate split dims and batch dim
    out_lbs = out_lbs.reshape(
        split_in_bounds.shape[0], 2, split_in_bounds.shape[2], *out_lbs.shape[1:]
    )
    out_ubs = out_ubs.reshape(
        split_in_bounds.shape[0], 2, split_in_bounds.shape[2], *out_ubs.shape[1:]
    )
    smaller_out_lb = torch.amin(out_lbs, dim=1)
    larger_out_ub = torch.amax(out_ubs, dim=1)
    out_lb_improve = smaller_out_lb - curr_best_out_lb
    out_ub_improve = curr_best_out_ub - larger_out_ub
    larger_improve = torch.maximum(out_lb_improve, out_ub_improve)
    # remove the network output dim
    larger_improve = torch.amax(larger_improve.flatten(start_dim=2), dim=2)
    split_dims = torch.argmax(larger_improve, dim=0)
    return split_dims


def split_longest_edge(branches: BranchStore) -> torch.Tensor:
    """
    Select dimensions to split.
    :code:`split_longest_edge` selects the dimension (for each batch element)
    that has the largest distance between lower and upper bound (the longest edge).

    :param branches: The branches for which to determine the dimensions to split.
    :return: A tensor of dimensions to split at their mid-point.
     The dimensions refer to a flattened input.
    """
    edge_len = branches.in_ubs - branches.in_lbs
    return torch.argmax(edge_len.flatten(1), dim=1)
