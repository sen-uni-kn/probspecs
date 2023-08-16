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
    def cdf(self, x: torch.Tensor):
        """
        The cumulative distribution function of the probability distribution.
        Supports batch processing.

        :param x: Where to evaluate the cumulative distribution function.
          This may be a batched tensor.
        :return: Cumulative distribution function evaluated at x.
        """
        raise NotImplementedError()
