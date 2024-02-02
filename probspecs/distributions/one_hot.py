#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
from scipy.stats import multinomial
import torch

from .probability_distribution import ProbabilityDistribution


class CategoricalOneHot(ProbabilityDistribution):
    """
    A categorical distribution (aka. generalized Bernoulli, aka. Multinoulli)
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

    def __init__(self, probabilities: torch.Tensor):
        """
        Create a new :code:`CategoricalOneHot` distribution.

        :param probabilities: The probability of each category, as a one-dimensional
         tensor.
         The entries of :code:`probability` must lie in [0.0, 1.0]
         and must sum to one.
        """
        if not torch.all((0.0 <= probabilities) & (probabilities <= 1.0)):
            raise ValueError(
                f"All entries of probabilities must lie in [0.0, 1.0]. "
                f"Got: {probabilities}"
            )
        if not torch.isclose(torch.sum(probabilities), torch.ones(())):
            raise ValueError(f"probabilities must sum to one. Got: {probabilities}")

        self.__probabilities = probabilities.flatten()
        # Each row of weighted_values is a one-hot encoded vector
        # It has shape (1, n, n), as this is useful when computing probabilities
        self.__values = torch.eye(self.__probabilities.size(0)).unsqueeze(0)
        self.__multinomial = multinomial(n=1, p=probabilities.detach().numpy())

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # shape of a and b is (N, n), where N is the batch size and n is the number
        # of categories / the size of self.__probabilities
        a, b = event
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        # value_contained has shape (N, n, n), where entry (i, j, k)
        # determines whether in batch element i one-hot vector with 1.0 at position k
        # is contained in [a, b] in dimension j.
        value_contained = (a <= self.__values) & (self.__values <= b)
        # a one-hot vector is only contained in event if it is contained
        # in all dimensions => reduce the j-dimension (see last comment)
        value_contained = value_contained.all(dim=-2)
        # weight contained values with their probabilities and sum over values
        value_probs = self.__probabilities.to(value_contained.device)
        value_contained = value_contained.to(value_probs.dtype)
        prob = torch.sum(value_contained * value_probs, dim=-1)
        return prob

    @property
    def event_shape(self) -> torch.Size:
        return self.__probabilities.shape

    # MARK: scipy-like methods

    def rvs(self, size: int = 1) -> torch.Tensor:
        values = self.__multinomial.rvs(size=size)
        # values are ints, we want floats.
        return torch.as_tensor(values, dtype=torch.get_default_dtype())

    def pmf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.__multinomial.pmf(x))

    def logpmf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.__multinomial.logpmf(x))

    def entropy(self) -> torch.Tensor:
        return torch.as_tensor(self.__multinomial.entropy())

    def cov(self) -> torch.Tensor:
        return torch.as_tensor(self.__multinomial.cov())
