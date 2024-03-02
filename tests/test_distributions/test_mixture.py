#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, truncnorm, beta

from probspecs import distributions
from probspecs.distributions import MixtureModel, UnivariateContinuousDistribution

import pytest


@pytest.fixture
def mixture_model_1():
    return MixtureModel(
        weights=[0.7, 0.3],
        distributions=(
            UnivariateContinuousDistribution(norm(loc=-10, scale=1)),
            UnivariateContinuousDistribution(norm(loc=10, scale=1)),
        ),
    )


def test_from_gaussian_mixture():
    np.random.seed(94485665)
    data = np.random.lognormal(0, 1, size=(1000, 1))
    data += np.random.normal(10, 2, size=(1000, 1))

    gmm = GaussianMixture(n_components=3)
    gmm.fit(data)

    distribution = MixtureModel.from_gaussian_mixture(gmm)
    print(distribution.weights, distribution.distributions)


def test_fit_truncnorm_mixture():
    rng = np.random.default_rng(303370415504141)
    data = np.vstack(
        [
            rng.normal(loc=-2.0, scale=1.0, size=(3000, 1)),
            rng.normal(loc=3.0, scale=1.5, size=(6000, 1)),
            rng.normal(loc=-20.0, scale=7.0, size=(1000, 1)),
        ]
    )
    mixture = MixtureModel.fit_truncnorm_mixture(
        data,
        bounds=(-22.0, 10.0),
        n_components=3,
        n_init=1,
        seed=rng.integers(0, 2**32, size=1).item(),
    )
    print(mixture.weights)


def test_fit_truncnorm_mixture_highly_concentrated():
    rng = np.random.default_rng(2370415204141)
    data = np.vstack(
        [
            np.zeros(100).reshape(-1, 1),
            rng.normal(loc=-2.0, scale=1.0, size=(100, 1)),
            rng.normal(loc=-2.0, scale=1.0, size=(100, 1)),
        ]
    )
    mixture = MixtureModel.fit_truncnorm_mixture(
        data,
        bounds=(-5.0, 5.0),
        n_components=3,
        n_init=1,
        seed=rng.integers(0, 2**32, size=1).item(),
    )
    print(mixture.weights)


def test_fit_truncnorm_mixture_too_many_components():
    rng = np.random.default_rng(303370415504141)
    data = np.vstack(
        [
            rng.normal(loc=-2.0, scale=1.0, size=(3000, 1)),
            rng.normal(loc=3.0, scale=1.5, size=(6000, 1)),
            rng.normal(loc=-20.0, scale=7.0, size=(1000, 1)),
        ]
    )
    mixture = MixtureModel.fit_truncnorm_mixture(
        data,
        bounds=(-22.0, 10.0),
        n_components=15,
        n_init=1,
        seed=rng.integers(0, 2**32, size=1).item(),
    )
    print(mixture.weights)


def test_fit_mixture_1():
    rng = np.random.default_rng(60713454971938)
    data = np.vstack(
        [
            rng.normal(loc=-2.0, scale=1.0, size=(3000, 1)),
            rng.normal(loc=3.0, scale=1.5, size=(6000, 1)),
            rng.normal(loc=-20.0, scale=7.0, size=(1000, 1)),
        ]
    )
    mixture = MixtureModel.fit_mixture(
        data,
        ((norm, (("loc", -20.0, 3.0), ("scale", 0.1, 10.0)), {}),) * 3,
        n_init=1,
        seed=rng.integers(0, 2**32, size=1).item(),
    )
    print(mixture.weights)


def test_fit_mixture_2():
    rng = np.random.default_rng(420640267505409)
    data = np.vstack(
        [
            rng.normal(loc=-2.0, scale=1.0, size=(3000, 1)),
            rng.normal(loc=3.0, scale=1.5, size=(6000, 1)),
            rng.normal(loc=-20.0, scale=7.0, size=(1000, 1)),
        ]
    )
    mixture = MixtureModel.fit_mixture(
        data,
        (
            (norm, (("loc", -20.0, 3.0), ("scale", 0.1, 10.0)), {}),
            (
                beta,
                (("a", None, None), ("b", None, None)),
                {"loc": -25.0, "scale": 30.0},
            ),
        ),
        n_init=1,
        seed=rng.integers(0, 2**32, size=1).item(),
    )
    print(mixture.weights)


@pytest.mark.parametrize(
    "event,expected_probability",
    [
        (([-20.0], [20.0]), 1.0),
        (([-20.0], [0.0]), 0.7),
        (([0.0], [20.0]), 0.3),
        (([-10.0], [0.0]), 0.35),
        (([0.0], 10.0), 0.15),
        (([-10.0], [10.0]), 0.5),
        (
            (
                [-20.0, -20.0, 0.0, -10.0, 0.0, -10.0],
                [20.0, 0.0, 20.0, 0.0, 10.0, 10.0],
            ),
            [1.0, 0.7, 0.3, 0.35, 0.15, 0.5],
        ),
    ],
)
def test_probability_1(mixture_model_1, event, expected_probability):
    dtype = mixture_model_1.dtype
    event = torch.tensor(event[0], dtype=dtype), torch.tensor(event[1], dtype=dtype)
    expected_probability = torch.tensor(expected_probability, dtype=dtype)
    assert torch.allclose(mixture_model_1.probability(event), expected_probability)


@pytest.mark.parametrize("batch_size", [1, 100])
def test_sample(batch_size):
    # create mixture model so that results lie in [-15, -5] or [5, 15]
    mixture_model = MixtureModel(
        weights=[0.85, 0.15],
        distributions=(
            distributions.wrap(truncnorm(a=-10, b=10, loc=-10, scale=0.5)),
            distributions.wrap(truncnorm(a=-10, b=10, loc=10, scale=0.5)),
        ),
    )

    x = mixture_model.sample(num_samples=batch_size, seed=675384213988767)
    assert torch.all(((-15.0 <= x) & (x <= -5.0)) | ((5.0 <= x) & (x <= 15.0)))


if __name__ == "__main__":
    pytest.main()
