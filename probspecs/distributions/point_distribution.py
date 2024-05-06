#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import torch

from .probability_distribution import ProbabilityDistribution, DiscreteDistribution
from ..utils.tensor_utils import TENSOR_LIKE, to_tensor


__all__ = ["PointDistribution"]


class PointDistribution(DiscreteDistribution):
    """
    A distribution that centers all probability mass on a single point.
    Useful for modelling fixed inputs.
    """

    def __init__(
        self,
        point: TENSOR_LIKE,
        dtype: torch.dtype = torch.double,
    ):
        """
        Creates a new :class:`PointDistribution`.

        :param point: The point having all probability mass.
        :param dtype: The floating point type that this distribution uses for
         sampling and computing probabilities.
        """
        self.__point = to_tensor(point, dtype=dtype)

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        event_lbs, event_ubs = event
        event_lbs, event_ubs = event_lbs.to(self.dtype), event_ubs.to(self.dtype)
        event_lbs = event_lbs.reshape((-1,) + self.event_shape)
        event_ubs = event_ubs.reshape((-1,) + self.event_shape)
        event_lbs = event_lbs.flatten(start_dim=1)
        event_ubs = event_ubs.flatten(start_dim=1)
        point = self.__point.flatten().unsqueeze(0)
        contained = (event_lbs <= point) & (event_ubs >= point)
        return torch.all(contained, dim=-1).float()

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        point = self.__point.unsqueeze(0)
        return point.expand(num_samples, *self.event_shape)

    @property
    def event_shape(self) -> torch.Size:
        return self.__point.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.__point.dtype

    @dtype.setter
    def dtype(self, dtype: torch):
        self.__point = self.__point.to(dtype)

    @property
    def parameters(self) -> torch.Tensor:
        return torch.empty((), dtype=self.dtype)

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        pass
