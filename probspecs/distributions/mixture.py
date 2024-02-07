# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from functools import reduce, partial
from typing import Sequence

import numpy as np
from scipy.stats import norm, truncnorm
import sklearn.mixture
import torch

from .probability_distribution import ProbabilityDistribution
from .single_dimension import UnivariateContinuousDistribution


class MixtureModel(ProbabilityDistribution):
    """
    A probability distribution that is represented by a mixture model,
    for example, a Gaussian mixture model.

    When sampling a mixture model, the mixture model first randomly selects
    one distribution from a fixed finite set of distributions.
    The selected distribution is then sampled to produce the sample of
    the mixture model.
    The probabilities with which the different distributions are selected
    are called the *mixture weights*.
    The distributions that are selected are called the *mixture components*.
    In a Gaussian mixture model, the mixture components are Gaussian/normal
    distributions.

    Read more on mixture models at https://en.wikipedia.org/wiki/Mixture_distribution.
    """

    def __init__(
        self,
        weights: Sequence[float] | np.ndarray | torch.Tensor,
        distributions: Sequence[ProbabilityDistribution],
    ):
        """
        Creates a new :code:`MixtureModel`.

        :param weights: The weights of the individual mixture components.
        :param distributions: The mixture components.
         All mixture components need to have the same event shape.
        """
        if len(distributions) == 0:
            raise ValueError("MixtureModel requires at least one component.")
        event_shape = distributions[0].event_shape
        for distribution in distributions[1:]:
            if distribution.event_shape != event_shape:
                raise ValueError(
                    f"Shape mismatch: all distributions must have the same shape. "
                    f"Got {event_shape} and {distribution.event_shape}"
                )
        self.__event_shape = event_shape

        if isinstance(weights, np.ndarray | torch.Tensor):
            weights = torch.as_tensor(weights)
        else:
            weights = torch.tensor(weights)
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional.")
        if not torch.isclose(torch.sum(weights), torch.ones((), dtype=weights.dtype)):
            raise ValueError("weights must sum to one.")
        if weights.size(0) != len(distributions):
            raise ValueError(
                "weights and distributions must have the same number of elements."
            )

        self.__weights = weights
        self.__distributions = tuple(distributions)

    @staticmethod
    def from_gaussian_mixture(
        mixture: sklearn.mixture.GaussianMixture,
        bounds: tuple[torch.Tensor | float, torch.Tensor | float] | None = None,
    ) -> "MixtureModel":
        """
        Create a univariante (1d) :code:`MixtureModel` distribution from a
        univariate sklearn Gaussian mixture model.

        Optionally, the Gaussian distributions in the mixture model can be
        truncated (see `scipy.stats.truncnorm`) to a certain range.
        This ensures that the total probability mass within the bounds is :math:`1.0`.

        :param mixture: The Gaussian mixture model.
         This needs to be a 1d model, which reflects in :code:`mixture.means_`
         and :code:`mixture.covariances_`.
         In particular, :code:`mixture.means_` needs to have a second dimension
         of size 1.
        :param bounds: Optional bounds to truncate the Gaussian distributions
         in the mixture model to. If :code:`None`, the distributions are not truncated.
        :return: A :code:`MixtureModel1d` behaving like :code:`mixture`.
        """
        n_components, n_features = mixture.means_.shape
        if n_features != 1:
            raise ValueError("mixture must be one-dimensional.")

        match mixture.covariance_type:
            case "spherical" | "full" | "diag":
                covariances = mixture.covariances_.reshape((n_components,))
            case "tied":
                covariances = np.repeat(mixture.covariances_, n_components)
            case _:
                raise NotImplementedError(
                    "Unknown mixture covariance type. Supported: "
                    "'spherical', 'full', 'diag', and 'tied'."
                )
        make_scipy_distr = norm
        if bounds is not None:
            lb, ub = bounds

            def make_truncnorm(loc, scale):
                a = (lb - loc) / scale
                b = (ub - loc) / scale
                return truncnorm(a, b, loc, scale)

            make_scipy_distr = make_truncnorm
        distributions = (
            make_scipy_distr(loc=mixture.means_[i, 0], scale=covariances[i])
            for i in range(n_components)
        )
        distributions = tuple(
            UnivariateContinuousDistribution(d) for d in distributions
        )
        return MixtureModel(mixture.weights_, distributions)

    @property
    def weights(self) -> torch.Tensor:
        return self.__weights.detach()

    @property
    def distributions(self) -> tuple[ProbabilityDistribution, ...]:
        return self.__distributions

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        component_probs = (
            component.probability(event) for component in self.__distributions
        )
        weighted = (w * p for w, p in zip(self.weights, component_probs))
        return sum(weighted)

    @property
    def event_shape(self) -> torch.Size:
        return self.__event_shape
