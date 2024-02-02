from math import prod

import torch

from .probability_distribution import ProbabilityDistribution


class MultivariateIndependent(ProbabilityDistribution):
    """
    Wraps several independent probability distributions into a
    multivariate probability distribution.

    This class flattens all inputs, such that it can work with arbitrarily
    shaped inputs.
    Similarly, it also flattens the elementary events of all underlying
    probability distributions.
    This means that you can combine, for example, a distribution with
    shape `(10,)` and another distribution with shape `(3, 32, 32)`
    into a single :code:`MultivariateIndependent` distribution with
    shape `(3082,)` or one with shape `(23, 67, 2)`.
    """

    def __init__(
        self,
        *distributions: ProbabilityDistribution,
        event_shape: tuple[int, ...] | torch.Size,
    ):
        """
        Creates a new :class:`MultidimensionalIndependent` object.

        :param distributions: The probability distributions of the different
         dimensions.
        :param event_shape: The shape of the elementary events supplied.
         This shape is used, in particular, for determining whether an event
         supplied to the :code:`probability` method is batched.
        """
        self.__num_elements = tuple(prod(d.event_shape) for d in distributions)
        if sum(self.__num_elements) != prod(event_shape):
            raise ValueError(
                f"The event shapes of the distributions are not compatible with "
                f"{event_shape}. While {event_shape} has {prod(event_shape)} elements, "
                f"the distributions overall have {sum(self.__num_elements)} elements."
            )

        self.__distributions = distributions
        self.__event_shape = torch.Size(event_shape)

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        lower_left, upper_right = event
        # add batch dimension if not already present
        lower_left = lower_left.reshape(-1, *self.event_shape).flatten(1)
        upper_right = upper_right.reshape(-1, *self.event_shape).flatten(1)

        i = 0
        prob = torch.ones(lower_left.size(0), device=lower_left.device)
        for distribution, num_elements in zip(
            self.__distributions, self.__num_elements
        ):
            lb = lower_left[:, i : i + num_elements]
            ub = upper_right[:, i : i + num_elements]
            i += num_elements

            lb = lb.reshape(-1, *distribution.event_shape)
            ub = ub.reshape(-1, *distribution.event_shape)
            prob = prob * distribution.probability((lb, ub))
        return prob

    @property
    def event_shape(self) -> torch.Size:
        return self.__event_shape
