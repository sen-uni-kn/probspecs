# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from random import Random
from typing import Sequence
from warnings import warn

import numpy as np
import scipy.stats
from scipy.stats import norm, truncnorm, multinomial
import sklearn.mixture
import torch

from .probability_distribution import ProbabilityDistribution
from .univariate import UnivariateContinuousDistribution
from ..utils.tensor_utils import TENSOR_LIKE, to_tensor


__all__ = ["MixtureModel"]


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
        dtype: torch.dtype = torch.double,
    ):
        """
        Creates a new :code:`MixtureModel`.

        :param weights: The weights of the individual mixture components.
        :param distributions: The mixture components.
         All mixture components need to have the same event shape.
        :param dtype: The floating point type that this distribution uses for
         sampling and computing probabilities.
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

        weights = to_tensor(weights).to(dtype)
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional.")
        if not torch.isclose(torch.sum(weights), torch.ones((), dtype=dtype)):
            raise ValueError("weights must sum to one.")
        if weights.size(0) != len(distributions):
            raise ValueError(
                "weights and distributions must have the same number of elements."
            )

        self.__weights = weights
        self.__distributions = tuple(distributions)

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
        component_probs = (p.to(self.dtype) for p in component_probs)
        weighted = (w * p for w, p in zip(self.weights, component_probs))
        return sum(weighted)

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        rng = torch.Generator()
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)

        # first generate random samples from all components,
        # then select using a sample from self.__select_component
        samples = []
        for component in self.__distributions:
            seed = torch.randint(2**63 - 1, size=(1,), generator=rng).item()
            sample = component.sample(num_samples, seed)
            sample = sample.to(self.dtype)
            samples.append(sample.reshape(num_samples, -1))
        samples = torch.hstack(samples)
        select_index = torch.multinomial(
            self.__weights, num_samples, replacement=True, generator=rng
        )
        return samples[torch.arange(samples.size(0)), select_index]

    @property
    def event_shape(self) -> torch.Size:
        return self.__event_shape

    @property
    def dtype(self) -> torch.dtype:
        return self.__weights.dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self.__weights = self.__weights.to(dtype)

    @staticmethod
    def from_gaussian_mixture(
        mixture: sklearn.mixture.GaussianMixture,
        dtype: torch.dtype = torch.double,
    ) -> "MixtureModel":
        """
        Create a univariante (1d) :code:`MixtureModel` distribution from a
        univariate sklearn Gaussian mixture model.

        :param mixture: The Gaussian mixture model.
         This needs to be a 1d model, which reflects in :code:`mixture.means_`
         and :code:`mixture.covariances_`.
         In particular, :code:`mixture.means_` needs to have a second dimension
         of size 1.
        :param dtype: The floating point type that the created mixture uses for
         sampling and computing probabilities.
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
        distributions = (
            norm(loc=mixture.means_[i, 0], scale=covariances[i])
            for i in range(n_components)
        )
        distributions = tuple(
            UnivariateContinuousDistribution(d) for d in distributions
        )
        return MixtureModel(mixture.weights_, distributions, dtype)

    @staticmethod
    def fit_truncnorm_mixture(
        data: TENSOR_LIKE,
        bounds: tuple[torch.Tensor | float, torch.Tensor | float],
        n_components=1,
        n_init=1,
        tolerance=1e-3,
        max_iters=100,
        seed: int | None = None,
        dtype: torch.dtype = torch.double,
    ) -> "MixtureModel":
        """
        Fit a mixture of truncated normal distributions (:code:`scipy.stats.truncnorm`)
        to a dataset.
        This class method can only fit one-dimensional data.

        This method uses expectation maximization to determine the parameters of the
        mixture using an iterative algorithm.
        It initializes the mixture parameters using k-means clustering.

        :param data: The one-dimensional data to fit.
        :param bounds: The lower and upper bounds where the normal distributions are to
         be truncated.
        :param n_components: The number of components of the fitted mixture model.
        :param n_init: The number restarts to perform.
         The mixture achieving the highest likelihood of the data is returned.
        :param tolerance: A tolerance parameter for determining convergence of
         expectation maximization.
        :param max_iters: The maximum number of expectation maximization iterations
         to perform.
        :param seed: A seed for initializing the random number generator used
         for :code:`kmeans` initialization.
        :param dtype: The floating point type that the created mixture uses for
         sampling and computing probabilities.
        """
        data = to_tensor(data)
        if data.ndim not in (1, 2):
            raise ValueError("Data neets to be one-dimensional.")
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        else:
            if data.size(-1) != 1:
                raise ValueError("Data neets to be one-dimensional.")
        data = data.detach().numpy()

        if n_init < 1:
            raise ValueError("n_init needs to be at least 1.")

        lb, ub = bounds
        if isinstance(lb, torch.Tensor):
            lb = lb.item()
        if isinstance(ub, torch.Tensor):
            ub = ub.item()
        # drop out-of-bounds data
        out_of_bounds = (data[:, 0] < lb) | (data[:, 0] > ub)
        data = data[~out_of_bounds, :]

        # Some components may start concentrating on a single data point
        # To avoid floating point problems in this case, limit how small
        # the scales can get.
        scale_eps = 1e-6

        best_log_likelihood = -torch.inf
        best_params = None
        rng = np.random.default_rng(seed)
        for _ in range(n_init):
            seed = rng.integers(0, 2**32)
            kmeans = sklearn.cluster.MiniBatchKMeans(
                n_components, random_state=seed, n_init=1, max_iter=50
            )
            cluster_assignment = kmeans.fit_predict(data)

            locs = kmeans.cluster_centers_
            scales = np.empty((n_components,))
            for i in range(n_components):
                cluster_data = data[cluster_assignment == i]
                if len(cluster_data) == 0:
                    scales[i] = 1.0
                else:
                    scales[i] = cluster_data.std()
            # if a cluster contains only a single data point, the variance is 0.0
            scales = np.clip(scales, a_min=scale_eps, a_max=None)
            weights = np.full((n_components,), fill_value=1 / n_components)

            last_log_likelihood = -np.inf
            for i in range(max_iters):
                # E-Step
                responsibilities = []
                for i in range(n_components):
                    loc = locs[i]
                    scale = scales[i]
                    a, b = (lb - loc) / scale, (ub - loc) / scale
                    pdf = truncnorm.pdf(data, a, b, loc, scale)
                    responsibilities.append(weights[i] * pdf)
                responsibilities = np.hstack(responsibilities)

                # data points on the boundary may still receive responsibilities of 0.0
                # if all distributions move away from the boundary (or scales get small)
                # due to floating point issues with bound conversion, etc.
                # We want to avoid computing the log of these 0.0 values.
                likelihoods = responsibilities.sum(axis=1)
                this_log_likelihood = np.log(
                    np.where(likelihoods > 0, likelihoods, 1.0)
                ).sum()  # log(1.0) = 0.0
                if np.abs(this_log_likelihood - last_log_likelihood) < tolerance:
                    last_log_likelihood = this_log_likelihood
                    break
                last_log_likelihood = this_log_likelihood

                # as discussed for the log likelihood above, responsibilities for
                # values on the boundaries be zero even for all components.
                # We also shouldn't divide these responsibilities by 0.0
                z = np.sum(responsibilities, axis=1).reshape(-1, 1)
                z = np.where(z > 0, z, 1.0)
                responsibilities = responsibilities / z

                # M-Step
                weights = responsibilities.mean(axis=0)
                total_resp = responsibilities.sum(axis=0)
                locs = (responsibilities * data).sum(axis=0) / total_resp
                square_dev = np.square(data - locs)
                variances = (responsibilities * square_dev).sum(axis=0) / total_resp
                scales = np.sqrt(variances)
                # Some components may start concentrating on a single data point
                # To avoid floating point problems in this case, limit how small
                # the scales can get.
                scales = np.clip(scales, a_min=scale_eps, a_max=None)

            if last_log_likelihood > best_log_likelihood:
                best_log_likelihood = last_log_likelihood
                best_params = {"weights": weights, "locs": locs, "scales": scales}

        # renormalize the weights so that they certainly sum to one
        weights = best_params["weights"] / best_params["weights"].sum()
        distributions = [
            UnivariateContinuousDistribution(
                truncnorm(
                    a=(lb - best_params["locs"][i]) / best_params["scales"][i],
                    b=(ub - best_params["locs"][i]) / best_params["scales"][i],
                    loc=best_params["locs"][i],
                    scale=best_params["scales"][i],
                )
            )
            for i in range(n_components)
        ]
        return MixtureModel(weights, distributions, dtype)

    @staticmethod
    def fit_mixture(
        data: TENSOR_LIKE,
        distributions: Sequence[
            tuple[
                type,
                tuple[tuple[str, float | None, float | None], ...],
                dict[str, float],
            ]
        ],
        n_init=1,
        seed: int | None = None,
        dtype: torch.dtype = torch.double,
        **minimize_kwargs,
    ) -> "MixtureModel":
        """
        Fit a mixture of arbitrary continuous `scipy` distributions
        (:code:`scipy.stats.rv_continuous`) to a dataset.

        This method uses `scipy.optimize.minimize` and `scipy.stats.rv_continuous.fit`
        to determine the mixture weights and the parameters of the mixtures.
        The mixture weights and all distribution parameters are initialized randomly.

        :param data: The data to fit.
        :param distributions: The distributions to fit and mix.
         This is a sequence of tuples consisting of
          - The distribution class for creating distributions.
            This class needs to have a `pdf` class method, for example
            :class`scipy.stats.norm`.
            The instances of these classes are wrapped as
            :class:`UnivariateContinuousDistributions`.
          - A sequence of tuples of keywords and bounds of the parameters to fit.
            The first element, the keyword, is used as keyword argument to create
            new distributions and evaluating the pdf using the distribution class.
            The second and third element of the tuple are the lower and upper bounds
            of this parameter for :code:`scipy.optimize.minimize`.
            Use `None` if the parameter is unbounded.
          - A dictionary of further keyword arguments for the initializer
            of the distribution class.
        :param n_init: The number restarts to perform.
         The mixture achieving the highest likelihood of the data is returned.
        :param seed: A seed for initializing the random number generator used
         for the random initialization.
        :param dtype: The floating point type that the created mixture uses for
        sampling and computing probabilities.
        :param minimize_kwargs: Further arguments for :code:`scipy.optimize.minimize`.
        :return: A mixture of the fitted `distributions`.
        """
        data = to_tensor(data)
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        data = data.detach().numpy()

        if n_init < 1:
            raise ValueError("n_init needs to be at least 1.")

        # maximize log-likelihood => minimize -log-likelihood
        def neg_log_likelihood(params):
            weights = params[: len(distributions)]
            i = len(weights)
            likelihoods = np.zeros((data.shape[0], 1))
            for w, (d_class, param_kws, further_kwargs) in zip(weights, distributions):
                param_kws = [kw for kw, _, _ in param_kws]
                d_params = params[i : i + len(param_kws)]
                d_params = dict(zip(param_kws, d_params))
                i += len(param_kws)
                pdf = d_class.pdf(data, **d_params, **further_kwargs)
                likelihoods += w * pdf

            # Treat likelihoods 0f 0.0, see fit_truncnorm_mixture.
            log_likelihood = np.log(np.where(likelihoods > 0, likelihoods, 1.0)).sum()
            return -log_likelihood

        best_log_likelihood = -torch.inf
        best_params = None
        bounds = [(0.0, 1.0)] * len(distributions) + [
            (lb, ub) for _, param_kws, _ in distributions for _, lb, ub in param_kws
        ]

        def normalized_weights(params):
            weights = params[: len(distributions)]
            # sum(weights) = 1.0
            return np.sum(weights) - 1.0

        rng = np.random.default_rng(seed)
        for _ in range(n_init):
            init_weights = rng.random(len(distributions))
            init_weights = init_weights / np.sum(init_weights)
            init_params = []
            for _, param_kws, _ in distributions:
                for _, lb, ub in param_kws:
                    lb = data.min() if lb is None else lb
                    ub = data.max() if ub is None else ub
                    param = rng.uniform(lb, ub)
                    init_params.append(param)
            init_params = np.array(init_params)
            init_params = np.hstack([init_weights, init_params])

            res = scipy.optimize.minimize(
                neg_log_likelihood,
                init_params,
                bounds=bounds,
                constraints={"type": "eq", "fun": normalized_weights},
                **minimize_kwargs,
            )
            if not res.success:
                warn(f"Optimization was not successful: {res.message}")
            if hasattr(res, "x"):
                log_likelihood = -neg_log_likelihood(res.x)
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_params = res.x

        if best_params is None:
            raise RuntimeError(
                "Fitting the mixture failed due to optimization failing."
            )

        weights = best_params[: len(distributions)]
        # optimization will not exactly satisfy the equality constraint sum(weights) = 1.0
        weights = weights / np.sum(weights)
        d_instances = []
        i = len(weights)
        for d_class, param_kws, further_kwargs in distributions:
            param_kws = [kw for kw, _, _ in param_kws]
            d_params = best_params[i : i + len(param_kws)]
            d_params = dict(zip(param_kws, d_params))
            i += len(param_kws)
            d = d_class(**d_params, **further_kwargs)
            d_instances.append(UnivariateContinuousDistribution(d))
        return MixtureModel(weights, d_instances, dtype)
