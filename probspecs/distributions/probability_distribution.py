# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from abc import abstractmethod
from typing import Protocol, runtime_checkable

import torch


__all__ = ["ProbabilityDistribution", "UnivariateDistribution"]


@runtime_checkable
class ProbabilityDistribution(Protocol):
    """
    A (potentially multivariate) discrete, continuous, or mixed
    probability distribution.
    """

    @abstractmethod
    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Computes the probability mass within
        a hyper-rectanglar set (an event).
        Supports batch processing (event may be a batch of hyper-rectangles).

        Implementation Node: Make sure to handle empty events properly.

        :param event: The hyper-rectangle whose probability mass is to be computed.
          The hyper-rectangle is provided as the "bottom-left" corner
          and the "top-right" corner, or, more generally, the smallest element
          of the hyper-rectangle in all dimensions and the largest element of
          the hyper-rectangle in all dimensions.
          The :code:`event` may also be a batch of hyper-rectangles.
          Generally, expect both the lower-left corner tensor and the
          upper-right corner tensors to have a batch dimension.
        :return: The probability of the :code:`event`.
         If :code:`event` is batched, returns a vector of probabilities.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        """
        Produce random samples from this probability distribution.

        :param num_samples: The number of samples to produce.
        :param seed: A seed to initializing random number generators.
        :return: A tensor with batch size (first dimension) :code:`num_samples`.
         The remaining dimensions of the tensor correspond to the :code:`event_shape`.
        """
        raise NotImplementedError()

    @property
    def event_shape(self) -> torch.Size:
        """
        The tensor shape of the elementary events underlying this
        probability distribution.

        :return: The shape of an elementary event.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """
        The floating point type (float, double, half) of this distribution.
        This is the dtype of the tensors returned by :code:`probability`
        and :code:`sample`.
        """
        raise NotImplementedError()

    @dtype.setter
    @abstractmethod
    def dtype(self, dtype: torch.dtype):
        """
        Sets the floating point type of this distribution.

        :param dtype: The new floating point type to use.
        """
        raise NotImplementedError()


class UnivariateDistribution(ProbabilityDistribution, Protocol):
    """
    A univariate (one-dimensional/single variable) probability distribution.
    """

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size((1,))
