from math import prod

import torch

from .probability_distribution import ProbabilityDistribution


class MultidimensionalIndependent(ProbabilityDistribution):
    """
    Wraps several independent 1d probability distributions into a
    multivariate probability distribution.

    This class flattens all inputs, such that it can work with arbitrarily
    shaped inputs.
    """

    def __init__(self, *distributions, input_shape):
        """
        Creates a new :class:`MultidimensionalIndependent` object.

        :param distributions: The probability distributions of the different
         dimensions.
        :param input_shape: The shape of the inputs supplied to the probability
         method. The input shape is used for determining whether the input is batched.
        """
        self.__distributions = distributions
        self.__input_shape = input_shape

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        lower_left, upper_right = event
        # add batch dimension if not already present
        lower_left = lower_left.reshape(-1, *self.__input_shape).flatten(1)
        upper_right = upper_right.reshape(-1, *self.__input_shape).flatten(1)

        probs = (
            self.__distributions[i].probability((lower_left[:, i], upper_right[:, i]))
            for i in range(lower_left.size(1))
        )
        return prod(probs)
