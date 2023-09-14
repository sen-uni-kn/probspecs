# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Generator

from frozendict import frozendict
import torch
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


def refine_bounds(
    network: nn.Module,
    input_bounds: tuple[torch.Tensor, torch.Tensor],
    auto_lirpa_method: str = "alpha-CROWN",
    auto_lirpa_bound_ops=frozendict(
        {"optimize_bound_args": frozendict({"iteration": 20, "lr_alpha": 0.1})}
    ),
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
    :param auto_lirpa_method: The :code:`auto_LiRPA` bound propagation method
     to use for computing bounds.
     More details in the :func:`auto_LiRPA.BoundedModule.compute_bounds` documentation.
    :param auto_lirpa_bound_ops: :code:`auto_LiRPA` bound propagation options.
     More details in the :func:`auto_LiRPA.BoundedModule` documentation.
    :return: A generator that yields improving lower and upper bounds.
    """
    initial_in_lb, initial_in_ub = input_bounds
    network = BoundedModule(network, initial_in_lb, auto_lirpa_bound_ops)
    bounded_tensor = construct_bounded_tensor(input_bounds)

    out_lb, out_ub = network.compute_bounds(
        x=(bounded_tensor,), method=auto_lirpa_method
    )
    yield (out_lb, out_ub)

    # TODO: refine


def construct_bounded_tensor(
    input_bounds: tuple[torch.Tensor, torch.Tensor]
) -> BoundedTensor:
    in_lb, in_ub = input_bounds
    input_domain = PerturbationLpNorm(x_L=in_lb, x_U=in_ub)
    midpoint = (in_ub + in_lb) / 2
    return BoundedTensor(midpoint, ptb=input_domain)
