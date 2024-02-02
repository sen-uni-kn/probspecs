#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
from scipy.stats import norm, bernoulli
import torch
import pytest

from probspecs import (
    MultivariateIndependent,
    UnivariateContinuousDistribution,
    UnivariateDiscreteDistribution,
    CategoricalOneHot,
)


def test_several_univariates():
    d1 = UnivariateContinuousDistribution(norm(loc=0, scale=1))
    d2 = UnivariateContinuousDistribution(norm(loc=0, scale=1))
    d3 = UnivariateDiscreteDistribution(bernoulli(p=0.33))
    multivariate = MultivariateIndependent(d1, d2, d3, event_shape=(3,))

    midpoint = torch.rand((100, 3)) * 10 - 5
    ranges = torch.rand((100, 3)) * 5
    lb = midpoint - ranges
    ub = midpoint + ranges
    p1 = d1.probability((lb[:, 0], ub[:, 0]))
    p2 = d2.probability((lb[:, 1], ub[:, 1]))
    p3 = d3.probability((lb[:, 2], ub[:, 2]))
    p_multivariate = multivariate.probability((lb, ub))
    assert torch.allclose(p_multivariate, p1 * p2 * p3)


def test_matrix_event():
    d1 = UnivariateContinuousDistribution(norm(loc=0, scale=1))
    d2 = UnivariateDiscreteDistribution(bernoulli(p=0.33))
    d3 = UnivariateDiscreteDistribution(bernoulli(p=0.75))
    d4 = UnivariateContinuousDistribution(norm(loc=3, scale=2))
    multivariate = MultivariateIndependent(d1, d2, d3, d4, event_shape=(2, 2))

    midpoint = torch.rand((100, 2, 2)) * 10 - 5
    ranges = torch.rand((100, 2, 2)) * 5
    lb = midpoint - ranges
    ub = midpoint + ranges
    p1 = d1.probability((lb[:, 0, 0], ub[:, 0, 0]))
    p2 = d2.probability((lb[:, 0, 1], ub[:, 0, 1]))
    p3 = d3.probability((lb[:, 1, 0], ub[:, 1, 0]))
    p4 = d4.probability((lb[:, 1, 1], ub[:, 1, 1]))
    p_multivariate = multivariate.probability((lb, ub))
    assert torch.allclose(p_multivariate, p1 * p2 * p3 * p4)


def test_stack_multivariates_1():
    p1 = torch.tensor([0.1, 0.7, 0.1, 0.1])
    p2 = torch.tensor([0.05, 0.1, 0.15, 0.3, 0.3, 0.1])
    d1 = CategoricalOneHot(p1)
    d2 = CategoricalOneHot(p2)
    multivariate = MultivariateIndependent(d1, d2, event_shape=(10,))

    assert torch.isclose(
        multivariate.probability((torch.zeros(10), torch.ones(10))), torch.ones(())
    )


def test_stack_multivariates_2():
    # fix the value of the first distributions
    p1 = torch.tensor([0.1, 0.7, 0.1, 0.1])
    p2 = torch.tensor([0.05, 0.1, 0.15, 0.3, 0.3, 0.1])
    d1 = CategoricalOneHot(p1)
    d2 = CategoricalOneHot(p2)
    multivariate = MultivariateIndependent(d1, d2, event_shape=(10,))

    n = 100
    value1 = torch.randint(4, (n,))
    expected_prob = p1[value1]
    value1 = torch.eye(4)[value1]
    lb = torch.hstack([value1, torch.zeros(n, 6)])
    ub = torch.hstack([value1, torch.ones(n, 6)])
    assert torch.allclose(multivariate.probability((lb, ub)), expected_prob)


def test_stack_multivariates_3():
    # fix the value of the second distribution
    p1 = torch.tensor([0.1, 0.7, 0.1, 0.1])
    p2 = torch.tensor([0.05, 0.1, 0.15, 0.3, 0.3, 0.1])
    d1 = CategoricalOneHot(p1)
    d2 = CategoricalOneHot(p2)
    multivariate = MultivariateIndependent(d1, d2, event_shape=(10,))

    n = 100
    value2 = torch.randint(6, (n,))
    expected_prob = p2[value2]
    value2 = torch.eye(6)[value2]
    lb = torch.hstack([torch.zeros(n, 4), value2])
    ub = torch.hstack([torch.ones(n, 4), value2])
    assert torch.allclose(multivariate.probability((lb, ub)), expected_prob)


def test_stack_multivariates_4():
    # fix values of both distributions
    p1 = torch.tensor([0.1, 0.7, 0.1, 0.1])
    p2 = torch.tensor([0.05, 0.1, 0.15, 0.3, 0.3, 0.1])
    d1 = CategoricalOneHot(p1)
    d2 = CategoricalOneHot(p2)
    multivariate = MultivariateIndependent(d1, d2, event_shape=(10,))

    n = 100
    value1 = torch.randint(4, (n,))
    value2 = torch.randint(6, (n,))
    expected_prob1 = p1[value1]
    expected_prob2 = p2[value2]
    expected_prob = expected_prob1 * expected_prob2
    value1 = torch.eye(4)[value1]
    value2 = torch.eye(6)[value2]
    lb = torch.hstack([value1, value2])
    ub = torch.hstack([value1, value2])
    assert torch.allclose(multivariate.probability((lb, ub)), expected_prob)


def test_stack_multivariates_5():
    # restrict both distributions to two values
    p1 = torch.tensor([0.1, 0.7, 0.1, 0.1])
    p2 = torch.tensor([0.05, 0.1, 0.15, 0.3, 0.3, 0.1])
    d1 = CategoricalOneHot(p1)
    d2 = CategoricalOneHot(p2)
    multivariate = MultivariateIndependent(d1, d2, event_shape=(10,))

    n = 100
    value11 = torch.randint(4, (n,))
    value12 = torch.randint(1, 4, (n,))
    value12 = (value11 + value12) % 4
    assert torch.all(value11 != value12)
    value21 = torch.randint(6, (n,))
    value22 = torch.randint(1, 6, (n,))
    value22 = (value21 + value22) % 6
    assert torch.all(value21 != value22)
    expected_prob1 = p1[value11] + p1[value12]
    expected_prob2 = p2[value21] + p2[value22]
    expected_prob = expected_prob1 * expected_prob2
    value11 = torch.eye(4)[value11]
    value12 = torch.eye(4)[value12]
    value21 = torch.eye(6)[value21]
    value22 = torch.eye(6)[value22]
    value1 = value11 + value12
    value2 = value21 + value22
    lb = torch.zeros(n, 10)
    ub = torch.hstack([value1, value2])
    assert torch.allclose(multivariate.probability((lb, ub)), expected_prob)
