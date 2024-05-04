#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import torch

from .categorical import Categorical
from ..utils.tensor_utils import TENSOR_LIKE

__all__ = ["CategoricalOneHot"]


class CategoricalOneHot(Categorical):
    """
    A categorical distribution (aka. generalized Bernoulli or Multinoulli)
    that produces one-hot vectors.

    For example, when the categories 1, 2, and 3 have probabilities 0.2, 0.5,
    and 0.3, respectively, a :code:`CategoricalOneHot` distribution with these
    probabilities will produce
    :code:`tensor([1.0, 0.0, 0.0])` with probability 0.2,
    :code:`tensor([0.0, 1.0, 0.0])` with probability 0.5,
    and :code:`tensor([0.0, 0.0, 1.0])` with probability 0.3.

    This class provides a few additional methods that are named following
    the :code:`scipy.stats` API.
    In particular, there are :code:`rvs` for drawing random samples
    from the categorical distribution, :code:`pmf` for evaluating
    the probability mass function, :code:`logpmf`, :code:`entropy`
    and :code:`cov`.
    For more details, see :code:`scipy.stats.multinomial`
    (a categorical distribution is a Multinomial distribution with n=1).
    """

    def __init__(
        self,
        probabilities: TENSOR_LIKE,
        frozen: bool = False,
        dtype: torch.dtype = torch.double,
    ):
        """
        Create a new :code:`CategoricalOneHot` distribution.

        :param probabilities: The probability of each category, as a one-dimensional
         tensor.
         The entries of :code:`probability` must lie in [0.0, 1.0]
         and must sum to one.
        :param frozen: Whether to allow optimizing the probabilities as parameters.
        :param dtype: The floating point that this distribution uses for
         sampling and computing probabilities.
        """
        super().__init__(probabilities, values=None, frozen=frozen, dtype=dtype)
        # Each row of weighted_values is a one-hot encoded vector
        # It has shape (1, n, n), as this is useful when computing probabilities
        self.__values = torch.eye(self.category_probabilities.size(0), dtype=dtype)
        self.__values.unsqueeze_(0)

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # shape of a and b is (N, n), where N is the batch size and n is the number
        # of categories / the size of self.category_probabilities
        a, b = event
        a = a.unsqueeze(-1).to(self.dtype)
        b = b.unsqueeze(-1).to(self.dtype)
        values = self.__values.to(a.device)
        # value_contained has shape (N, n, n), where entry (i, j, k)
        # determines whether in batch element i one-hot vector with 1.0 at position k
        # is contained in [a, b] in dimension j.
        value_contained = (a <= values) & (values <= b)
        # a one-hot vector is only contained in event if it is contained
        # in all dimensions => reduce the j-dimension (see last comment)
        value_contained = value_contained.all(dim=-2)
        # weight contained values with their probabilities and sum over values
        value_probs = self.category_probabilities.to(a.device)
        prob = torch.sum(value_contained * value_probs, dim=-1)
        return prob

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        idx = super().sample(num_samples, seed)
        return self.__values[0][idx]

    @property
    def event_shape(self) -> torch.Size:
        return self.category_probabilities.shape

    @property
    def dtype(self) -> torch.dtype:
        return super(CategoricalOneHot, self).dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        super(CategoricalOneHot, self.__class__).dtype.fset(self, dtype)
        self.__values = self.__values.to(dtype)
