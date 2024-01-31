#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

from probspecs.distributions import MixtureModel1d, ContinuousDistribution1d

import pytest


@pytest.fixture
def mixture_model_1():
    return MixtureModel1d(
        weights=[0.7, 0.3],
        distributions=(
            ContinuousDistribution1d(norm(loc=-10, scale=1)),
            ContinuousDistribution1d(norm(loc=10, scale=1)),
        ),
    )


def test_from_gaussian_mixture():
    np.random.seed(94485665)
    data = np.random.lognormal(0, 1, size=(1000, 1))
    data += np.random.normal(10, 2, size=(1000, 1))

    gmm = GaussianMixture(n_components=3)
    gmm.fit(data)

    distribution = MixtureModel1d.from_gaussian_mixture(gmm)
    print(distribution.weights, distribution.distributions)


@pytest.mark.parametrize(
    "event,probability_bounds",
    [
        ((torch.tensor(-20.0), torch.tensor(20.0)), (0.999, 1.001)),
        ((torch.tensor(-20.0), torch.tensor(0.0)), (0.699, 0.701)),
        ((torch.tensor(0.0), torch.tensor(20.0)), (0.299, 0.391)),
        ((torch.tensor(-10.0), torch.tensor(0.0)), (0.349, 0.351)),
        ((torch.tensor(0.0), torch.tensor(10.0)), (0.149, 0.151)),
        ((torch.tensor(-10.0), torch.tensor(10.0)), (0.499, 0.501)),
    ],
)
def test_probability_1(mixture_model_1, event, probability_bounds):
    lower, upper = probability_bounds
    assert lower <= mixture_model_1.probability(event) <= upper


def test_probability_1_batched(mixture_model_1):
    event_lb = torch.tensor([-20.0, -20.0, 0.0, -10.0, 0.0, -10.0])
    event_ub = torch.tensor([20.0, 0.0, 20.0, 0.0, 10.0, 10.0])
    expected_prob = torch.tensor([1.0, 0.7, 0.3, 0.35, 0.15, 0.5], dtype=torch.double)
    assert torch.allclose(
        mixture_model_1.probability((event_lb, event_ub)), expected_prob
    )
