from .probability_distribution import ProbabilityDistribution
from .as_integer import AsInteger
from .bayesian_network import BayesianNetwork
from .categorical import Categorical
from .mixture import MixtureModel
from .multivariate import MultivariateIndependent
from .one_hot import CategoricalOneHot
from .point_distribution import PointDistribution
from .uniform import Uniform
from .univariate import (
    UnivariateContinuousDistribution,
    UnivariateDiscreteDistribution,
)


def wrap(
    distribution,
) -> UnivariateContinuousDistribution | UnivariateDiscreteDistribution:
    """
    Wraps a :code:`scipy.stats` distribution as a
    :code:`UnivariateContinuousDistribution` or
    :code:`UnivariateDiscreteDistribution` object.
    It determines if the distribution is discrete by checking if
    it has a `pmf` method.

    You are responsible for ensuring that :code:`distribution` is
    univariate.

    :param distribution: The :code:`scipy.stats` distribution to wrap.
    :return: A :code:`ProbabilityDistribution` object wrapping :code:`distribution`.
    """
    if hasattr(distribution, "pmf"):
        return UnivariateDiscreteDistribution(distribution)
    else:
        return UnivariateContinuousDistribution(distribution)
