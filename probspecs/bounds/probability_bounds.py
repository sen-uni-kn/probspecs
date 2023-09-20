# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Generator, Literal

from ..formula import Probability


def probability_bounds(
    probability: Probability,
    variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 128,
)
