# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch

from .probability_distribution import DiscreteDistribution, ProbabilityDistribution
from .univariate import UnivariateContinuousDistribution


__all__ = ["AsInteger"]


class AsInteger(DiscreteDistribution):
    """
    Wraps a univariate continuous distribution as an integer distribution.
    It assumes that the values of the continuous distribution are rounded
    the nearest integer.
    Concretely, values are assumed to be rounded using :code:`torch.round`.

    :code:`AsInteger` rounds values, but does not change the floating point
    type of the underlying distribution.
    """

    def __init__(self, distribution: ProbabilityDistribution):
        if distribution.event_shape != (1,):
            raise ValueError("Can only wrap univariate distributions.")
        self.__distribution = distribution

    @classmethod
    def wrap(cls, distribution) -> "AsInteger":
        """
        Wraps a univariate :class:`ProbabilityDistribution` or an object with a
        :code:`cdf` and :code:`rvs` method (for example, a :code:`scipy.stats`
        distribution) as a ordinal distribution.
        """
        if (
            hasattr(distribution, "cdf")
            and hasattr(distribution, "pdf")
            and hasattr(distribution, "rvs")
        ):
            distribution = UnivariateContinuousDistribution(distribution)
            return AsInteger(distribution)
        elif isinstance(distribution, ProbabilityDistribution):
            return AsInteger(distribution)
        else:
            raise ValueError(f"Can not wrap object: {distribution}")

    @property
    def continuous_distribution(self) -> ProbabilityDistribution:
        return self.__distribution

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = event
        # Only the integer values matter.
        # Convert the interval to the range in which the values are rounded to the
        # smallest / largest integer in the interval.
        # Note that to where we round N.5 (to zero/to inf/to even/..., N is some integer)
        # doesn't matter because N.5 itself has zero probability since our distribution
        # is continuous
        a = a.ceil() - 0.5
        b = b.floor() + 0.5
        return self.__distribution.probability((a, b))

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        samples = self.__distribution.sample(num_samples, seed)
        return torch.round(samples)

    @property
    def event_shape(self) -> torch.Size:
        return self.__distribution.event_shape

    @property
    def dtype(self) -> torch.dtype:
        return self.__distribution.dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self.__distribution.dtype = dtype

    @property
    def parameters(self) -> torch.Tensor:
        return self.__distribution.parameters

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        self.__distribution.parameters = parameters

    @property
    def _parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__distribution._parameter_bounds
