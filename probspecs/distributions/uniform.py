#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import torch

from .probability_distribution import ProbabilityDistribution
from ..utils.tensor_utils import TENSOR_LIKE, to_tensor


__all__ = ["Uniform"]


class Uniform(ProbabilityDistribution):
    """
    A univariate or multivariate uniform distribution.
    """

    def __init__(
        self,
        support: tuple[TENSOR_LIKE, TENSOR_LIKE],
        dtype: torch.dtype = torch.double,
    ):
        """
        Creates a new :class:`Uniform` distribution.

        :param support: The hyper-rectangular region in which the
         the uniform distribution has non-zero probability.
         The maximal values of the support are excluded from the uniform
         distribution (see :code:`torch.rand`).
        :param dtype: The floating point type that this distribution uses for
         sampling and computing probabilities.
        """
        lbs, ubs = support
        lbs, ubs = to_tensor(lbs), to_tensor(ubs)
        self.__lbs, self.__ubs = lbs.to(dtype), ubs.to(dtype)
        self.__range = self.__ubs - self.__lbs
        self.__total_volume = torch.prod(self.__range)

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        event_lbs, event_ubs = event
        event_lbs, event_ubs = event_lbs.to(self.dtype), event_ubs.to(self.dtype)
        event_lbs = event_lbs.reshape((-1,) + self.event_shape)
        event_ubs = event_ubs.reshape((-1,) + self.event_shape)
        intersection_lbs = torch.maximum(self.__lbs, event_lbs)
        intersection_ubs = torch.minimum(self.__ubs, event_ubs)
        intersection_range = (intersection_ubs - intersection_lbs).flatten(1)
        intersection_volume = torch.prod(intersection_range, dim=1)
        return intersection_volume / self.__total_volume

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        rng = torch.Generator(device=self.__lbs.device)
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)
        x = torch.rand(
            (num_samples,) + self.event_shape, generator=rng, dtype=self.__lbs.dtype
        )
        return self.__lbs + x * self.__range

    @property
    def event_shape(self) -> torch.Size:
        return self.__lbs.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.__lbs.dtype

    @dtype.setter
    def dtype(self, dtype: torch):
        self.__lbs = self.__lbs.to(dtype)
        self.__ubs = self.__ubs.to(dtype)
        self.__range = self.__range.to(dtype)
        self.__total_volume = self.__total_volume.to(dtype)
