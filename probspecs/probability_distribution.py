# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from abc import abstractmethod
from math import prod
from typing import Protocol, runtime_checkable

import torch
import numpy as np
import scipy.integrate
from frozendict import frozendict


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


class ContinuousDistribution1d(ProbabilityDistribution):
    """
    Wraps a continuous 1d probability distribution that allows evaluating
    the cumulative distribution function (cdf) as a :class:`ProbabilityDistribution`.

    The probability of interval :math:`[a, b]` is computed
    as :math:`cdf(b) - cdf(a)`.

    If underlying probability distribution returns numpy arrays instead
    of tensors (for example, scipy.stats distributions), the result
    is wrapped as a tensor.
    Consequently, this class can be used to leverage scipy distributions.
    Example: :code:`Distribution1d(scipy.stats.norm)`
    """

    def __init__(self, distribution):
        self.__distribution = distribution

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = event
        orig_device = a.device
        a = a.detach().cpu()
        b = b.detach().cpu()
        cdf_high = self.__distribution.cdf(b)
        cdf_low = self.__distribution.cdf(a)
        prob = cdf_high - cdf_low
        prob = torch.as_tensor(prob, device=orig_device)
        return prob


class DiscreteDistribution1d(ProbabilityDistribution):
    """
    Wraps a discrete 1d probability distribution that allows provides a
    probability mass function (pmf) as a :class:`ProbabilityDistribution`.

    The probability of an interval :math:`[a, b]` is computed as the sum of
    the pmf of all integer values within :math:`[a, b]`.

    If underlying probability distribution returns numpy arrays instead
    of tensors (for example, scipy.stats distributions), the result
    is wrapped as a tensor.
    Consequently, this class can be used to leverage scipy distributions.
    Example: :code:`Distribution1d(scipy.stats.bernoulli)`
    """

    def __init__(self, distribution):
        self.__distribution = distribution

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = event
        min_ = torch.min(a).ceil()
        max_ = torch.min(b).floor()
        # Add 0.1 since arange excludes the end point
        integers = torch.arange(min_, max_ + 0.1, step=1)
        integers = integers.detach().cpu()
        probs = self.__distribution.pmf(integers)
        probs = torch.as_tensor(probs, device=a.device)
        # reshape a, b and integers for broadcasting
        a = a.reshape(-1, 1)
        b = a.reshape(-1, 1)
        integers = integers.reshape(1, -1).to(a.device)
        selected_probs = torch.where((a <= integers) & (b >= integers), probs, 0.0)
        return selected_probs.sum(dim=1)


class MultidimensionalIndependent(ProbabilityDistribution):
    """
    Wraps several independent 1d probability distributions into a
    multivariate probability distribution.

    This class flattens all inputs, such that it can work with arbitrarily
    shaped inputs.
    """

    def __init__(self, *distributions, input_shape):
        """
        Creates a new :class:`MultidimensionalIndependent` object.

        :param distributions: The probability distributions of the different
         dimensions.
        :param input_shape: The shape of the inputs supplied to the probability
         method. The input shape is used for determining whether the input is batched.
        """
        self.__distributions = distributions
        self.__input_shape = input_shape

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        lower_left, upper_right = event
        # add batch dimension if not already present
        lower_left = lower_left.reshape(-1, *self.__input_shape).flatten(1)
        upper_right = upper_right.reshape(-1, *self.__input_shape).flatten(1)

        probs = (
            self.__distributions[i].probability((lower_left[:, i], upper_right[:, i]))
            for i in range(lower_left.size(1))
        )
        return prod(probs)
