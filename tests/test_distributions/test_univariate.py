#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import torch
from scipy.stats import norm, lognorm

from probspecs.distributions import UnivariateContinuousDistribution

import pytest


@pytest.mark.parametrize(
    "distribution,event,expected_probability",
    [
        (
            UnivariateContinuousDistribution(norm(), (-1.0, 1.0)),
            (-1.0, 1.0),
            1.0,
        ),
        (
            UnivariateContinuousDistribution(norm(), (-1.0, 1.0)),
            (0.0, 1.0),
            0.5,
        ),
        (
            UnivariateContinuousDistribution(norm(), (-1.0, 1.0)),
            ([-1.0, 0.0], [1.0, 1.0]),
            [1.0, 0.5],
        ),
        (
            UnivariateContinuousDistribution(lognorm(s=1.0), (0.5, 1.0)),
            (0.5, 1.0),
            1.0,
        ),
    ],
)
def test_truncate(distribution, event, expected_probability):
    dtype = distribution.dtype
    event = torch.tensor(event[0], dtype=dtype), torch.tensor(event[1], dtype=dtype)
    expected_probability = torch.tensor(expected_probability, dtype=dtype)
    assert torch.allclose(distribution.probability(event), expected_probability)
