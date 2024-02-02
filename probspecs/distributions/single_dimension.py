import torch

from .probability_distribution import UnivarianteDistribution


class UnivariateContinuousDistribution(UnivarianteDistribution):
    """
    Wraps a continuous univariante (1d) probability distribution that
    allows evaluating the cumulative distribution function (cdf)
    as a :class:`ProbabilityDistribution`.

    The probability of interval :math:`[a, b]` is computed
    as :math:`cdf(b) - cdf(a)`.

    If underlying probability distribution returns numpy arrays instead
    of tensors (for example, scipy.stats distributions), the result
    is wrapped as a tensor.
    Consequently, this class can be used to leverage scipy distributions.
    Example: :code:`UnivariateContinuousDistribution(scipy.stats.norm)`
    """

    def __init__(self, distribution):
        self.__distribution = distribution

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = event
        orig_device = a.device
        a = a.detach().flatten().cpu()
        b = b.detach().flatten().cpu()
        cdf_high = self.__distribution.cdf(b)
        cdf_low = self.__distribution.cdf(a)
        prob = cdf_high - cdf_low
        prob = torch.as_tensor(prob, device=orig_device)
        return prob


class UnivariateDiscreteDistribution(UnivarianteDistribution):
    """
    Wraps a discrete univariate (1d) probability distribution that provides a
    probability mass function (pmf) as a :class:`ProbabilityDistribution`.

    The probability of an interval :math:`[a, b]` is computed as the sum of
    the pmf of all integer values within :math:`[a, b]`.

    If underlying probability distribution returns numpy arrays instead
    of tensors (for example, scipy.stats distributions), the result
    is wrapped as a tensor.
    Consequently, this class can be used to leverage scipy distributions.
    Example: :code:`UnivariateDiscreteDistribution(scipy.stats.bernoulli)`
    """

    def __init__(self, distribution):
        self.__distribution = distribution

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = event
        min_ = torch.min(a).ceil()
        max_ = torch.max(b).floor()
        # Add 0.1 since arange excludes the end point
        integers = torch.arange(min_, max_ + 0.1, step=1)
        integers = integers.detach().cpu()
        probs = self.__distribution.pmf(integers)
        probs = torch.as_tensor(probs, device=a.device)
        # reshape a, b and integers for broadcasting
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        integers = integers.reshape(1, -1).to(a.device)
        selected_probs = torch.where((a <= integers) & (integers <= b), probs, 0.0)
        return selected_probs.sum(dim=1)
