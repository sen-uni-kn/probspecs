# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
import torch.nn.functional as F


def clamp(input, min=None, max=None):
    """Implements :code:`torch.clamp` using ReLU, addition, and subtraction."""
    return minimum(maximum(input, min), max)


def maximum(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Implements :code:`torch.maximum` using ReLU, addition, and subtraction."""
    return F.relu(input - other) + other


def minimum(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Implements :code:`torch.minimum` using ReLU, addition, and subtraction."""
    return -F.relu(other - input) + other
