# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from scipy.stats import norm, lognorm, truncnorm
import torch
import pytest

from probspecs.distributions import AsInteger, UnivariateContinuousDistribution


def test_create_standard_normal():
    distribution = AsInteger.wrap(norm)
    return distribution


def test_create_truncnorm():
    loc = 1.0
    scale = 2.0
    a = -1.0
    b = 3.0
    a, b = (a - loc) / scale, (b - loc) / scale
    distribution = AsInteger.wrap(truncnorm(a, b, loc, scale))
    return distribution


def test_create_lognorm():
    distribution = UnivariateContinuousDistribution(
        lognorm(0.0, 1.0), bounds=(0.0, 5.0)
    )
    distribution = AsInteger.wrap(distribution)
    return distribution


integer_standard_normal = pytest.fixture(test_create_standard_normal)
integer_truncnorm = pytest.fixture(test_create_truncnorm)


@pytest.mark.parametrize(
    "distribution,event,expected_probability",
    [
        ("integer_standard_normal", ([-10.0], [10.0]), 1.0),
        ("integer_standard_normal", ([0.0], [0.0]), 0.38292492254802624),
        ("integer_standard_normal", ([1.0], [1.0]), 0.2417303374571288),
        ("integer_standard_normal", ([-1.0], [-1.0]), 0.2417303374571288),
        ("integer_standard_normal", ([2.0], [2.0]), 0.060597535943081926),
        ("integer_standard_normal", ([-2.0], [-2.0]), 0.060597535943081926),
        ("integer_standard_normal", ([3.0], [3.0]), 0.005977036246740619),
        ("integer_standard_normal", ([-3.0], [-3.0]), 0.005977036246740619),
        ("integer_standard_normal", ([20.0], [20.0]), 0.0),
        ("integer_standard_normal", ([-20.0], [-20.0]), 0.0),
        ("integer_standard_normal", ([0.0], [1.0]), 0.624655260005155),
        ("integer_standard_normal", ([-1.0], [1.0]), 0.8663855974622838),
        ("integer_standard_normal", ([0.1], [1.999]), 0.2417303374571288),
        ("integer_standard_normal", ([-1.999], [1.999]), 0.8663855974622838),
        (  # batched version of the above
            "integer_standard_normal",
            (
                [
                    -10.0,
                    0.0,
                    1.0,
                    -1.0,
                    2.0,
                    -2.0,
                    3.0,
                    -3.0,
                    20.0,
                    -20.0,
                    0.0,
                    -1.0,
                    0.1,
                    -1.999,
                ],
                [
                    10.0,
                    0.0,
                    1.0,
                    -1.0,
                    2.0,
                    -2.0,
                    3.0,
                    -3.0,
                    20.0,
                    -20.0,
                    1.0,
                    1.0,
                    1.999,
                    1.999,
                ],
            ),
            [
                1.0,
                0.38292492254802624,
                0.2417303374571288,
                0.2417303374571288,
                0.060597535943081926,
                0.060597535943081926,
                0.005977036246740619,
                0.005977036246740619,
                0.0,
                0.0,
                0.624655260005155,
                0.8663855974622838,
                0.2417303374571288,
                0.8663855974622838,
            ],
        ),
        ("integer_truncnorm", ([-1.0], [3.0]), 1.0),
        ("integer_truncnorm", ([-1.0], [-1.0]), 0.09956517454609119),
        ("integer_truncnorm", ([0.0], [0.0]), 0.25585031548300835),
        ("integer_truncnorm", ([1.0], [1.0]), 0.2891690199418012),
        ("integer_truncnorm", ([2.0], [2.0]), 0.2558503154830084),
        ("integer_truncnorm", ([3.0], [3.0]), 0.09956517454609119),
        ("integer_truncnorm", ([4.0], [4.0]), 0.0),
        ("integer_truncnorm", ([3.1], [5.0]), 0.0),
        ("integer_truncnorm", ([2.5], [3.5]), 0.09956517454609082),
        (
            "integer_truncnorm",
            (
                [-1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.1, 2.5],
                [3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.5],
            ),
            [
                1.0,
                0.09956517454609119,
                0.25585031548300835,
                0.2891690199418012,
                0.2558503154830084,
                0.09956517454609119,
                0.0,
                0.0,
                0.09956517454609119,
            ],
        ),
    ],
)
def test_probability(distribution, event, expected_probability, request):
    distribution = request.getfixturevalue(distribution)
    event = tuple(map(torch.tensor, event))
    prob = distribution.probability(event)
    assert torch.allclose(prob, torch.tensor(expected_probability, dtype=prob.dtype))


if __name__ == "__main__":
    pytest.main()
