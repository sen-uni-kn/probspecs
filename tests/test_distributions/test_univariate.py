#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import torch
from scipy.stats import norm, lognorm

from probspecs.distributions import UnivariateContinuousDistribution

import pytest


@pytest.mark.parametrize(
    "distribution,event,probability_bounds",
    [
        (
            UnivariateContinuousDistribution(norm(), (-1.0, 1.0)),
            (-1.0, 1.0),
            (0.999, 1.001),
        ),
        (
            UnivariateContinuousDistribution(norm(), (-1.0, 1.0)),
            (0.0, 1.0),
            (0.499, 0.501),
        ),
        (
            UnivariateContinuousDistribution(lognorm(s=1.0), (0.5, 1.0)),
            (0.5, 1.0),
            (0.999, 1.001),
        ),
    ],
)
def test_truncate(distribution, event, probability_bounds):
    event = torch.tensor(event[0]), torch.tensor(event[1])
    lower, upper = probability_bounds
    assert lower <= distribution.probability(event) <= upper
