#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import torch
import pytest

from probspecs.distributions import Uniform


@pytest.mark.parametrize(
    "support",
    [
        ([0.0], [1.0]),
        ([-5.0], [3.1415]),
        ([-100.0, 0.5], [100.0, 0.75]),
        (
            [
                [[1.23, 2.34, 3.45], [-1.23, -2.34, -3.45], [-10.0, -10.0, -22.0]],
                [[2.34, 3.45, 4.56], [1.23, 2.34, 7.021], [0.0, 10.0, -11.0]],
            ]
        ),
    ],
)
def test_sample_uniform(support):
    support = tuple(map(torch.tensor, support))
    lbs, ubs = support
    distribution = Uniform(support)
    x = distribution.sample(1000, seed=459407045812444)
    assert torch.all(lbs <= x)
    assert torch.all(x <= ubs)


@pytest.mark.parametrize(
    "support,event,expected_probability",
    [
        (([0.0], [1.0]), ([0.0], [1.0]), 1.0),
        (([0.0], [1.0]), ([0.0], [0.5]), 0.5),
        (([0.0], [1.0]), ([0.5], [1.0]), 0.5),
        (([0.0], [1.0]), ([-1.0], [0.0]), 0.0),
        (([0.0], [1.0]), ([-1.0], [2.0]), 1.0),
        (([0.0], [1.0]), ([-0.1], [0.1]), 0.1),
        (([-5.0], [3.1415]), ([-5.0], [3.1415]), 1.0),
        (([-5.0], [1.0]), ([-5.0], [0.0]), 0.83333333333),
        (([-5.0], [1.0]), ([-4.5], [0.5]), 0.83333333333),
        (([-5.0], [1.0]), ([-4.0], [1.0]), 0.83333333333),
        (([-5.0], [1.0]), ([-4.0], [1000.0]), 0.83333333333),
        (([-5.0], [1.0]), ([0.0], [1.0]), 0.16666666667),
        (([-100.0, 0.5], [100.0, 0.75]), ([-100.0, 0.5], [100.0, 0.75]), 1.0),
        (([-100.0, 0.5], [100.0, 0.75]), ([0.0, 0.5], [100.0, 0.5]), 0.0),
        (([-100.0, 0.5], [100.0, 0.75]), ([0.0, 0.625], [100.0, 0.75]), 0.25),
        (([-100.0, 0.5], [100.0, 0.75]), ([-25.0, 0.625], [75.0, 0.75]), 0.25),
        (([-100.0, 0.5], [100.0, 0.75]), ([-25.0, 0.5], [75.0, 0.625]), 0.25),
        (([-100.0, 0.5], [100.0, 0.75]), ([-33.5, 0.51], [66.5, 0.635]), 0.25),
    ],
)
def test_probability_uniform(support, event, expected_probability):
    support = tuple(map(torch.tensor, support))
    event = tuple(map(torch.tensor, event))
    distribution = Uniform(support)
    assert torch.isclose(
        distribution.probability(event),
        torch.tensor(expected_probability, dtype=distribution.dtype),
    )
