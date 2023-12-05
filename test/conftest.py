# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import scipy.stats
import torch
from torch import nn
import numpy as np
from pathlib import Path

from probspecs import TensorInputSpace
from probspecs.probability_distribution import (
    Distribution1d,
    MultidimensionalIndependent,
)

import pytest


@pytest.fixture
def resource_dir():
    preferred_path = Path("test/resources")
    if preferred_path.exists():
        return preferred_path
    else:
        return Path("resources")


@pytest.fixture
def verification_test_nets_1d():
    """
    A fixture supplying:
     - A 1d tensor input space with bounds [-10, 10]
     - A standard normal distribution for this input space
     - A binary classifier network for x >= 0.
       This means that output 0 of this network is greater-equal
       output 1, iff x >= 0.
     - A binary classifier network for x >= 1 (analogous to the first network)
    """
    # 1d input space of a normally distributed random variable
    input_space = TensorInputSpace(
        lbs=torch.tensor([-10.0]),
        ubs=torch.tensor([10.0]),
    )
    distribution = Distribution1d(scipy.stats.norm)

    # binary classifier
    # net produces the first class if the input is >= 0.0
    # and the second class otherwise
    net = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2))
    with torch.no_grad():
        net[0].weight.data = torch.tensor([[1.0], [-1.0]])
        net[0].bias.data = torch.zeros(2)
        net[2].weight.data = torch.eye(2)
        net[2].bias.data = torch.zeros(2)

    # binary classifier indicating if the input is >= 1.0
    net2 = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2))
    with torch.no_grad():
        net2[0].weight.data = torch.tensor([[1.0], [-1.0]])
        net2[0].bias.data = torch.tensor([-1.0, 1.0])
        net2[2].weight.data = torch.eye(2)
        net2[2].bias.data = torch.zeros(2)

    return net, net2, input_space, distribution


@pytest.fixture
def verification_test_compose():
    """
    A fixture supplying:
     - A 2d tensor input space with bounds :math:`[-10, 10]^2`
     - A combination of two independent standard normal distributions for the
       2d input space
     - A FCNN producing 100d outputs.
     - A FCNN consuming 100d inputs and prodicing a 1d output.
    """
    torch.manual_seed(554549888315709)

    input_space = TensorInputSpace(
        lbs=torch.full((2,), fill_value=-10.0),
        ubs=torch.full((2,), fill_value=10.0),
    )
    distrs = [Distribution1d(scipy.stats.norm) for i in range(2)]
    distribution = MultidimensionalIndependent(*distrs, input_shape=(2,))

    generator = nn.Sequential(
        nn.Linear(2, 10, bias=False),
        nn.ReLU(),
        nn.Linear(10, 100, bias=False),
    )
    consumer = nn.Sequential(nn.Linear(100, 10), nn.ReLU(), nn.Linear(10, 1))
    return input_space, distribution, generator, consumer


@pytest.fixture
def verification_test_mnist_gen(resource_dir):
    """
    A fixture supplying a
     - A 1d tensor input space with bounds [-3.0, 3.0]
     - A combination of independent normal distributions for a (1, 4, 1, 1) input space
     - An MNIST generator network trained as part of a GAN
    """
    input_space = TensorInputSpace(
        lbs=torch.full((4, 1, 1), fill_value=-3.0),
        ubs=torch.full((4, 1, 1), fill_value=3.0),
    )

    np.random.seed(128891471)
    means = np.random.rand(4) * 2 - 1
    stds = np.random.rand(4) * 1
    distrs = [Distribution1d(scipy.stats.norm(means[i], stds[i])) for i in range(4)]
    distribution = MultidimensionalIndependent(*distrs, input_shape=(4, 1, 1))

    generator = torch.load(resource_dir / "mnist_generator.pyt")
    return input_space, distribution, generator


@pytest.fixture
def small_conv_mnist_net(resource_dir):
    return torch.load(resource_dir / "small_conv_mnist_network.pyt")
