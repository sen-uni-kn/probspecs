# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch

from .probability_distribution import ProbabilityDistribution
from ..utils.tensor_utils import TENSOR_LIKE, to_tensor


__all__ = ["Categorical"]


class Categorical(ProbabilityDistribution):
    """
    A categorical distribution (aka. generalized Bernoulli or Multinoulli)
    produces one from a set of discrete values that each have a certain probability.

    If the vector of category probabilities has `n` values,
    :class:`Categorical` produces samples in :code:`[0, n-1]`.

    For a version of a categorical distribution that produces one-hot encoded vectors,
    see :code:`CategoricalOneHot.
    """

    def __init__(
        self,
        probabilities: TENSOR_LIKE,
        values: TENSOR_LIKE | None = None,
        dtype: torch.dtype = torch.double,
    ):
        """
        Create a new :code:`Categorical` distribution.

        :param probabilities: The probability of each category, as a one-dimensional
         tensor.
         The entries of :code:`probability` must lie in [0.0, 1.0]
         and must sum to one.
        :param values: The values of this :class:`Categorical` distribution.
         If :code:`None`, the values are taken to be
         :code:`torch.arange(0, len(probabilities))`.
        :param dtype: The floating point that this distribution uses for
         sampling and computing probabilities.
        """
        probabilities = to_tensor(probabilities, dtype).flatten()

        if not torch.all((0.0 <= probabilities) & (probabilities <= 1.0)):
            raise ValueError(
                f"All entries of probabilities must lie in [0.0, 1.0]. "
                f"Got: {probabilities}"
            )
        if not torch.isclose(torch.sum(probabilities), torch.ones((), dtype=dtype)):
            raise ValueError(f"probabilities must sum to one. Got: {probabilities}")

        if values is not None:
            values = to_tensor(values, dtype).flatten()
            values, sort_idx = torch.sort(values, stable=True)
            probabilities = probabilities[sort_idx]
        else:
            values = torch.arange(len(probabilities))

        self.__probabilities = probabilities
        self.__values = values

    @property
    def category_probabilities(self) -> torch.Tensor:
        """
        The probabilities of the different categories
        (values or classes of the categorical distribution).
        The order of :code:`categorical_probabilities` matches the order
        of :code:`values`.
        """
        return self.__probabilities

    @property
    def values(self) -> torch.Tensor:
        """
        The values this distribution can take on.
        The values are always sorted in ascending order.
        """
        return self.__values

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = event
        a, b = a.to(self.dtype), b.to(self.dtype)
        value_contained: torch.Tensor = (a <= self.__values) & (self.__values <= b)
        value_probs = self.__probabilities.to(value_contained.device)
        probs = torch.where(value_contained, value_probs, 0.0)
        return torch.sum(probs, dim=-1)

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        rng = torch.Generator()
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)
        idx = torch.multinomial(
            self.__probabilities, num_samples, replacement=True, generator=rng
        )
        return self.__values[idx]

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size((1,))

    @property
    def dtype(self) -> torch.dtype:
        return self.__probabilities.dtype

    @dtype.setter
    def dtype(self, dtype):
        self.__probabilities = self.__probabilities.to(dtype)
        self.__values = self.__values.to(dtype)
