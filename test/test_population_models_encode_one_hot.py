# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch

import pytest
from probspecs.population_models import EncodeOneHot


@pytest.fixture
def one_continuous_one_categorical():
    return (None, 3)


@pytest.fixture
def three_continuous_two_categorical():
    return (None, 5, None, 2, None)


@pytest.mark.parametrize("i,j", [(0, 0), (2, 6), (4, 9)])
def test_copy_continuous(three_continuous_two_categorical, i, j):
    """
    :param i: Index in purely numeric input space
    :param j: Index in one-hot encoded input space
    """
    torch.manual_seed(380085079305903)

    encode = EncodeOneHot(three_continuous_two_categorical)
    for _ in range(10):
        x = 100 * torch.rand(100, 5) - 50
        y = encode(x)
        assert torch.allclose(x[:, i], y[:, j])


@pytest.mark.parametrize("a", [0, 1, 2])
def test_one_hot_1(one_continuous_one_categorical, a):
    torch.manual_seed(578997103338228)

    encode = EncodeOneHot(one_continuous_one_categorical)
    batch_size = 100
    for _ in range(10):
        x = 100 * torch.rand(batch_size, 2) - 50
        x[:, 1] = a - 0.5 + torch.rand(batch_size)
        y = encode(x)
        assert torch.allclose(y[:, 1 + a], torch.ones(batch_size))
        for b in range(3):
            if b != a:
                assert torch.allclose(y[:, 1 + b], torch.zeros(batch_size))


def test_one_hot_2(one_continuous_one_categorical):
    torch.manual_seed(71997102338228)

    encode = EncodeOneHot(one_continuous_one_categorical)
    batch_size = 100
    for _ in range(10):
        x = 100 * torch.rand(batch_size, 2) - 50
        x[:, 1] = 0.5 - 10 * torch.rand(batch_size)
        y = encode(x)
        assert torch.allclose(y[:, 1], torch.ones(batch_size))
        assert torch.allclose(y[:, 2], torch.zeros(batch_size))
        assert torch.allclose(y[:, 3], torch.zeros(batch_size))


def test_one_hot_3(one_continuous_one_categorical):
    torch.manual_seed(11997102338918)

    encode = EncodeOneHot(one_continuous_one_categorical)
    batch_size = 100
    for _ in range(10):
        x = 100 * torch.rand(batch_size, 2) - 50
        x[:, 1] = 1.5 + 10 * torch.rand(batch_size)
        y = encode(x)
        assert torch.allclose(y[:, 1], torch.zeros(batch_size))
        assert torch.allclose(y[:, 2], torch.zeros(batch_size))
        assert torch.allclose(y[:, 3], torch.ones(batch_size))


@pytest.mark.parametrize("a", [0, 1])
def test_one_hot_binary(three_continuous_two_categorical, a):
    torch.manual_seed(455374230005877)

    encode = EncodeOneHot(three_continuous_two_categorical)
    batch_size = 100
    for _ in range(10):
        x = 100 * torch.rand(batch_size, 5) - 50
        x[:, 3] = a - 0.5 + torch.rand(batch_size)
        y = encode(x)
        assert torch.allclose(y[:, 7 + (1 - a)], torch.zeros(batch_size))
        assert torch.allclose(y[:, 7 + a], torch.ones(batch_size))
