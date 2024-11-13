# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import warnings
from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.optimize
import torch


__all__ = ["ProbabilityDistribution", "UnivariateDistribution", "DiscreteDistribution"]

from probspecs.utils.tensor_utils import TENSOR_LIKE, to_tensor


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

        Implementation Note: Make sure to handle empty events properly.

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
    def density(self, elementary: torch.Tensor) -> torch.Tensor:
        """
        Computes the probability density for an elementary event (a point).
        For discrete event spaces, we define the density to be equal to the probability
        mass at :code:`elementary`.
        Supports batch processing (`elementary` may be a batch of elementary events).

        :param elementary: The elementary event whose probability density is
          to be computed.
          The :code:`elementary` event may also be a batch of elementary events.
          Generally, :code:`elementary` has a batch dimension.
        :return: The probability density of the :code:`elementary` event.
         If :code:`elementary` is batched, returns a vector of densities.
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
    @abstractmethod
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

    @property
    def parameters(self) -> torch.Tensor:
        """
        The parameters of this probability distribution.
        Return an empty tensor (for example, :code:`torch.zeros(())`) if
        your distribution is parameter-free.
        """
        raise NotImplementedError()

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        """
        Sets the parameters of this probability distribution.

        :param parameters: The new parameter values. Needs to have the same
         shape as the value of the :code:`parameters` property.
        """
        raise NotImplementedError()

    @property
    def _parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The range of permitted values for the parameters of this distribution.
        For use by :code:`fit`.

        By default, this parameter is a tuple of -inf and +inf tensors with the
        shape of the :code:`parameters`.
        This sets the parameters to be unrestricted.
        """
        parameters = self.parameters
        all_inf = torch.full_like(parameters, fill_value=torch.inf)
        return (-all_inf, all_inf)

    def fit(self, data: TENSOR_LIKE, **minimize_kwargs) -> float:
        """
        Fits the parameters of this distribution to :code:`data` using
        maximum likelihood estimation.

        Uses :code:`scipy.optimize.minimize` for fitting the data.

        :param data: The data to which to fit the distribution.
        :param minimize_kwargs: Additional keyword arguments to pass
          to :code:`scipy.optimize.minimize`.
        :return:
        """
        data = to_tensor(data, self.dtype)

        def objective(parameters: np.array):
            self.parameters = to_tensor(parameters, self.dtype)
            likelihood = self.density(data).detach().numpy()
            # avoid calculating logarithms of points with 0.0 log likelihood
            # Assign these points small but positive probability.
            return -np.mean(np.log(np.where(likelihood > 0, likelihood, 1e-32)))

        initial_parameters = self.parameters.detach().numpy()
        param_lb, param_ub = self._parameter_bounds
        bounds = scipy.optimize.Bounds(
            param_lb.detach().numpy(), param_ub.detach().numpy(), keep_feasible=True
        )
        if len(minimize_kwargs) == 0:
            minimize_kwargs = {
                "method": "SLSQP",
                "options": {"eps": 1e-4},
            }
        res = scipy.optimize.minimize(
            objective,
            initial_parameters,
            bounds=bounds,
            **minimize_kwargs,
        )

        if hasattr(res, "x") and hasattr(res, "fun"):
            if not res.message:
                warnings.warn(
                    f"Fitting failed with message: {res.message}. Continuing with intermediate result."
                )
            parameters = to_tensor(res.x, self.dtype)
            self.parameters = parameters
            return float(res.fun)
        else:
            raise RuntimeError(f"Fitting failed with message: {res.message}.")


class UnivariateDistribution(ProbabilityDistribution, Protocol):
    """
    A univariate (one-dimensional/single variable) probability distribution.
    """

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size((1,))


class DiscreteDistribution(ProbabilityDistribution, Protocol):
    """
    A discrete probability distribution.

    This Protocol provides a default implementation for
    :code:`density(x)` that returns the value of :code:`probability((x, x))`.
    """

    def density(self, elementary: torch.Tensor) -> torch.Tensor:
        return self.probability((elementary, elementary))
