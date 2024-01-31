# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from abc import abstractmethod
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ProbabilityDistribution(Protocol):
    """
    A multivariate discrete, continuous, or mixed probability distribution.
    """

    @abstractmethod
    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Computes the probability mass within
        a hyper-rectanglar set (an event).
        Supports batch processing (event may be a batch of hyper-rectangles).

        :param event: The hyper-rectangle whose probability mass is to be computed.
          The hyper-rectangle is provided as the "bottom-left" corner
          and the "top-right" corner, or, more generally, the smallest element
          of the hyper-rectangle in all dimensions and the largest element of
          the hyper-rectangle in all dimensions.
          The :code:`event` may also be a batch of hyper-rectangles.
          Generally, expect both the lower-left corner tensor and the
          upper-right corner tensors to have a batch dimension-
        :return: The probability of the :code:`event`.
         If :code:`event` is batched, returns a vector of probabilities.
        """
        raise NotImplementedError()
