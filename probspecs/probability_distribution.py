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
    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        The cumulative distribution function of the probability distribution.
        Supports batch processing.

        :param x: Where to evaluate the cumulative distribution function.
          This may be a batched tensor.
        :return: Cumulative distribution function evaluated at x.
        """
        raise NotImplementedError()


class ToTensor(ProbabilityDistribution):
    """
    Wraps the values returned by another probability distribution in tensors.

    Can be used, for example to leverage scipy distributions.
    Example: :code:`ToTensor(scipy.stats.norm)`
    """

    def __init__(self, distribution):
        self.__distribution = distribution

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        value = self.__distribution.cdf(x)
        return torch.as_tensor(value)
