# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import scipy.stats
import torch
from torch import nn

from probspecs import TensorInputSpace
from probspecs import ToTensor

import pytest


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
    distribution = ToTensor(scipy.stats.norm)

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
