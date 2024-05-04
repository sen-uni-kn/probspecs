# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import torch

from .probability_distribution import UnivariateDistribution, DiscreteDistribution

__all__ = ["UnivariateContinuousDistribution", "UnivariateDiscreteDistribution"]


class UnivariateContinuousDistribution(UnivariateDistribution):
    """
    Wraps a continuous univariate (one-dimensional) probability distribution that
    allows evaluating the cumulative distribution function (cdf),
    evaluating the probability density function (pdf), and
    obtaining random samples (rvs) as a :class:`ProbabilityDistribution`.

    The method evaluating the cdf of the wrapped distribution
    needs to be named :code:`cdf`.
    The method evaluating the pdf of the wrapped distribution
    needs to be named :code:`pdf`.
    The method producing random samples needs to be named :code:`rvs`
    and accept a :code:`size` and a :code:`random_state` argument,
    both of type :code:`int`.

    If the distribution is parameterized (:code:`parameters` argument
    of the initializer), the `cdf`, `pdf`, and `rvs` methods also need to
    accept arguments for the parameters, in the order the parameters are
    given.

    The probability of interval :math:`[a, b]` is computed
    as :math:`cdf(b) - cdf(a)`.

    Optionally, this class allows to truncate the probability distribution
    to an interval.
    In this case, all probabilities are normalized by the total probability
    mass of the distribution in the base interval.
    Concretely, the probability of the inveral :math:`[a, b]` is computed
    as :math:`\\frac{cdf(b) - cdf(a)}{cdf(d) - cdf(c)}, where
    :math:`[c, d]` is the base interval to which we truncate the distribution.
    See :code:`scipy.stats.truncnorm` for an example of a truncated probability
    distribution.

    If underlying probability distribution returns numpy arrays instead
    of tensors (for example, scipy.stats distributions), the result
    is wrapped as a tensor.
    Consequently, this class can be used to leverage scipy distributions.
    Example: :code:`UnivariateContinuousDistribution(scipy.stats.norm)`
    """

    def __init__(
        self,
        distribution,
        parameters: tuple[float, ...] = (),
        bounds: tuple[float, float] | None = None,
        parameter_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        dtype: torch.dtype = torch.double,
    ):
        """
        Wraps :code:`distribution` as a :code:`UnivariateContinuousDistribution`.

        :param distribution: The distribution to wrap.
        :param parameters: The parameters of the distribution.
        :param bounds: The base interval for truncating :code:`distribution`.
         If :code:`bounds` is :code:`None`, the distribution isn't truncated.
        :param parameter_bounds: Lower and upper ounds of the :code:`parameters`
         to respect when fitting the parameters.
         All parameter are unbounded by default.
        :param dtype: The floating point `torch` data type
         the values returned by :code:`distribution` are cast to.
        """
        self.__distribution = distribution
        self.__parameters = parameters
        self.__param_bounds = parameter_bounds
        self.__total_mass = 1.0
        if bounds is not None:
            lb, ub = bounds
            self.__total_mass = distribution.cdf(ub) - distribution.cdf(lb)
        self.__dtype = dtype

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = event
        orig_device = a.device
        a = a.detach().flatten().cpu()
        b = b.detach().flatten().cpu()
        cdf_high = self.__distribution.cdf(b, *self.__parameters)
        cdf_low = self.__distribution.cdf(a, *self.__parameters)
        prob = (cdf_high - cdf_low) / self.__total_mass
        prob = torch.as_tensor(prob, device=orig_device, dtype=self.__dtype)
        # if a > b, cdf_high - cdf_low is negative, however the event is actually
        # empty, therefore we set prob = 0.0
        prob = torch.clamp(prob, min=0.0)
        return prob

    def density(self, elementary: torch.Tensor) -> torch.Tensor:
        orig_device = elementary.device
        elementary = elementary.detach().flatten().cpu()
        density = self.__distribution.pdf(elementary, *self.__parameters)
        density = density / self.__total_mass
        density = torch.as_tensor(density, device=orig_device, dtype=self.__dtype)
        return density

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        # this class is intended for use with scipy, which relies on numpy.random
        # which doesn't like seeds >= 2**32.
        if seed is not None:
            seed = seed % 2**32
        samples = self.__distribution.rvs(
            *self.__parameters, size=num_samples, random_state=seed
        )
        return torch.as_tensor(samples, dtype=self.__dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype):
        self.__dtype = dtype

    @property
    def parameters(self) -> torch.Tensor:
        return torch.tensor(self.__parameters, dtype=self.__dtype)

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        self.__parameters = tuple(parameters.tolist())

    @property
    def _parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.__param_bounds is None:
            return super()._parameter_bounds
        else:
            lb, ub = self.__param_bounds
            return torch.tensor(lb, dtype=self.dtype), torch.as_tensor(
                ub, dtype=self.dtype
            )


class UnivariateDiscreteDistribution(UnivariateDistribution, DiscreteDistribution):
    """
    Wraps a discrete univariate (1d) probability distribution that provides a
    probability mass function (pmf) and obtaining random samples (rvs)
    as a :class:`ProbabilityDistribution`.

    The method evaluating the pmf of the wrapped distribution
    needs to be named :code:`pmf`.
    The method producing random samples needs to be named :code:`rvs`
    and accept a :code:`size` and a :code:`random_state` argument,
    both of type :code:`int`.

    The probability of an interval :math:`[a, b]` is computed as the sum of
    the pmf of all integer values within :math:`[a, b]`.

    If underlying probability distribution returns numpy arrays instead
    of tensors (for example, scipy.stats distributions), the result
    is wrapped as a tensor.
    Consequently, this class can be used to leverage scipy distributions.
    Example: :code:`UnivariateDiscreteDistribution(scipy.stats.bernoulli)`
    """

    def __init__(
        self,
        distribution,
        parameters: tuple[float, ...] = (),
        parameter_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        dtype: torch.dtype = torch.double,
    ):
        """
        Wraps :code:`distribution` as a :code:`UnivariateDiscreteDistribution`.

        :param distribution: The distribution to wrap.
        :param parameters: The parameters of the distribution.
        :param parameter_bounds: Lower and upper bounds of the :code:`parameters`
         to respect when fitting the parameters.
         All parameter are unbounded by default.
        :param dtype: The floating point `torch` data type
         the values returned by :code:`distribution` are cast to.
        """
        self.__distribution = distribution
        self.__parameters = parameters
        self.__param_bounds = parameter_bounds
        self.__dtype = dtype

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = event
        min_ = torch.min(a).ceil()
        max_ = torch.max(b).floor()
        # Add 0.1 since arange excludes the end point
        integers = torch.arange(min_, max_ + 0.1, step=1, dtype=self.__dtype)
        integers = integers.detach().cpu()
        probs = self.__distribution.pmf(integers, *self.__parameters)
        probs = torch.as_tensor(probs, device=a.device, dtype=self.__dtype)
        # reshape a, b and integers for broadcasting
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        integers = integers.reshape(1, -1).to(a.device)
        selected_probs = torch.where((a <= integers) & (integers <= b), probs, 0.0)
        return selected_probs.sum(dim=1)

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        # this class is intended for use with scipy, which relies on numpy.random
        # which doesn't like seeds >= 2**32.
        if seed is not None:
            seed = seed % 2**32
        samples = self.__distribution.rvs(
            *self.parameters, size=num_samples, random_state=seed
        )
        return torch.as_tensor(samples, dtype=self.__dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype):
        self.__dtype = dtype

    @property
    def parameters(self) -> torch.Tensor:
        return torch.tensor(self.__parameters, dtype=self.__dtype)

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        self.__parameters = tuple(parameters.tolist())

    @property
    def _parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.__param_bounds is None:
            return super()._parameter_bounds
        else:
            lb, ub = self.__param_bounds
            return torch.tensor(lb, dtype=self.dtype), torch.as_tensor(
                ub, dtype=self.dtype
            )
