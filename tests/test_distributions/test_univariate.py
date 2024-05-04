#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import torch
from scipy.stats import beta, norm, lognorm, bernoulli

from conftest import check_fit, check_probabilities
from probspecs.distributions import (
    UnivariateContinuousDistribution,
    UnivariateDiscreteDistribution,
)

import pytest


@pytest.mark.parametrize(
    "distribution,event,expected_probability",
    [
        (
            UnivariateContinuousDistribution(norm()),
            (-1.0, 1.0),
            0.6826895,
        ),
        (
            UnivariateContinuousDistribution(norm()),
            (0.0, 2.0),
            0.4772499,
        ),
        (
            UnivariateContinuousDistribution(norm, parameters=(0.0, 1.0)),
            (-1.0, 1.0),
            0.6826895,
        ),
        (
            UnivariateContinuousDistribution(norm, parameters=(0.0, 1.0)),
            (0.0, 2.0),
            0.4772499,
        ),
        (
            UnivariateContinuousDistribution(norm, parameters=(10.0, 1.0)),
            (9.0, 11.0),
            0.6826895,
        ),
        (
            UnivariateContinuousDistribution(norm, parameters=(10.0, 1.0)),
            (10.0, 12.0),
            0.4772499,
        ),
        (
            UnivariateContinuousDistribution(norm, parameters=(1.0, 2.0)),
            (-1.0, 3.0),
            0.6826895,
        ),
        (
            UnivariateContinuousDistribution(norm, parameters=(1.0, 2.0)),
            (1.0, 5.0),
            0.4772499,
        ),
        (
            UnivariateDiscreteDistribution(bernoulli(0.75)),
            (1.0, 1.0),
            0.75,
        ),
        (
            UnivariateDiscreteDistribution(bernoulli(0.75)),
            (0.0, 0.0),
            0.25,
        ),
        (
            UnivariateDiscreteDistribution(bernoulli, parameters=(0.75,)),
            (1.0, 1.0),
            0.75,
        ),
        (
            UnivariateDiscreteDistribution(bernoulli, parameters=(0.75,)),
            (0.0, 0.0),
            0.25,
        ),
        (
            UnivariateDiscreteDistribution(bernoulli, parameters=(0.75, 3.0)),
            (4.0, 4.0),
            0.75,
        ),
        (
            UnivariateDiscreteDistribution(bernoulli, parameters=(0.75, 3.0)),
            (3.0, 3.0),
            0.25,
        ),
        (
            UnivariateContinuousDistribution(beta(a=2.31, b=0.627)),
            (0.0, 0.25),
            0.02129459,
        ),
        (
            UnivariateContinuousDistribution(beta(a=2.31, b=0.627)),
            (0.1, 0.6),
            0.18306679,
        ),
        (
            UnivariateContinuousDistribution(beta, parameters=(2.31, 0.627)),
            (0.0, 0.25),
            0.02129459,
        ),
        (
            UnivariateContinuousDistribution(beta, parameters=(2.31, 0.627)),
            (0.1, 0.6),
            0.18306679,
        ),
    ],
)
def test_parameters(distribution, event, expected_probability):
    check_probabilities(distribution, event, expected_probability)


@pytest.mark.parametrize(
    "distribution,event,expected_probability",
    [
        (
            UnivariateContinuousDistribution(norm(), bounds=(-1.0, 1.0)),
            (-1.0, 1.0),
            1.0,
        ),
        (
            UnivariateContinuousDistribution(norm(), bounds=(-1.0, 1.0)),
            (0.0, 1.0),
            0.5,
        ),
        (
            UnivariateContinuousDistribution(norm(), bounds=(-1.0, 1.0)),
            ([-1.0, 0.0], [1.0, 1.0]),
            [1.0, 0.5],
        ),
        (
            UnivariateContinuousDistribution(lognorm(s=1.0), bounds=(0.5, 1.0)),
            (0.5, 1.0),
            1.0,
        ),
    ],
)
def test_truncate(distribution, event, expected_probability):
    check_probabilities(distribution, event, expected_probability)


@pytest.mark.parametrize(
    "distribution,data,expected_parameters",
    [
        (
            UnivariateDiscreteDistribution(bernoulli, (0.1,)),
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.5],
        ),
        (
            UnivariateDiscreteDistribution(bernoulli, (0.0,)),
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.9],
        ),
    ],
)
def test_fit_bernoulli(distribution, data, expected_parameters):
    check_fit(distribution, data, expected_parameters)


def test_fit_normal():
    distribution = UnivariateContinuousDistribution(
        norm, parameters=(0.0, 1.0), parameter_bounds=((-10.0, -10.0), (10.0, 10.0))
    )
    data = (torch.randn(100) - 2.25) * 0.9
    check_fit(distribution, data, [-2.25, 0.9], atol=0.5)


def test_fit_lognorm():
    distribution = UnivariateContinuousDistribution(
        lognorm,
        parameters=(1.0, 0.0, 1.0),
        parameter_bounds=((0.000001, 0.0, 0.0), (torch.inf, torch.inf, torch.inf)),
    )
    data = (torch.randn(100)) * 3.0
    distribution.fit(data)
    print(distribution.parameters)
