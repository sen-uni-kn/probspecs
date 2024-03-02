# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm


__all__ = ["construct_bounded_tensor"]


def construct_bounded_tensor(in_lb: torch.Tensor, in_ub: torch.Tensor) -> BoundedTensor:
    input_domain = PerturbationLpNorm(x_L=in_lb, x_U=in_ub)
    midpoint = (in_ub + in_lb) / 2
    return BoundedTensor(midpoint, ptb=input_domain)
