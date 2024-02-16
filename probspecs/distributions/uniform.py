#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import torch

from .probability_distribution import ProbabilityDistribution


class Uniform(ProbabilityDistribution):
    """
    A univariate or multivariate uniform distribution.
    """

    def __init__(self, support: tuple[torch.Tensor, torch.Tensor]):
        """
        Creates a new :class:`Uniform` distribution.

        :param support: The hyper-rectangular region in which the
         the uniform distribution has non-zero probability.
         The maximal values of the support are excluded from the uniform
         distribution (see :code:`torch.rand`).
        """
        self.__lbs, self.__ubs = support
        self.__range = self.__ubs - self.__lbs
        self.__total_volume = torch.prod(self.__range)

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        event_lbs, event_ubs = event
        event_lbs = event_lbs.reshape((-1,) + self.event_shape)
        event_ubs = event_ubs.reshape((-1,) + self.event_shape)
        intersection_lbs = torch.maximum(self.__lbs, event_lbs)
        intersection_ubs = torch.minimum(self.__ubs, event_ubs)
        intersection_range = (intersection_ubs - intersection_lbs).flatten(1)
        intersection_volume = torch.prod(intersection_range, dim=1)
        return intersection_volume / self.__total_volume

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        rng = None
        if seed is not None:
            rng = torch.Generator(device=self.__lbs.device)
            rng.manual_seed(seed)
        x = torch.rand(
            (num_samples,) + self.event_shape, generator=rng, dtype=self.__lbs.dtype
        )
        return self.__lbs + x * self.__range

    @property
    def event_shape(self) -> torch.Size:
        return self.__lbs.shape
