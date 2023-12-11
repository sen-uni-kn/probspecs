# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F


class FairSquareV2BaseNN(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        # N_age = ((age - 17.0) / 73.0 - 0.5) * 10 + 0.5
        # N_education_num = ((education_num - 3.0) / 13.0 - 0.5) * 10 + 0.5
        self.normalize_offset = torch.tensor(
            [
                17.0 + 0.5 * 73 - 0.5 * 7.3,  # age
                3.0 + 0.5 * 13.0 - 0.5 * 1.3,  # education_num
            ]
        )
        self.normalize_scale = torch.tensor([7.3, 1.3])
        # auto_LiRPA has some problem if these are buffers
        # self.register_buffer("normalize_offset", normalize_offset)
        # self.register_buffer("normalize_scale", normalize_scale)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()


class FairSquareNNV2H1(FairSquareV2BaseNN):
    """
    Since neuron neural network model from
    https://github.com/sedrews/fairsquare/blob/master/oopsla/noqual/M_ind_F_NN_V2_H1.fr
    """

    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.tensor([[0.1718, 1.1416]]))
        self.bias1 = nn.Parameter(torch.tensor([0.4754]))
        self.weight2 = nn.Parameter(torch.tensor([[0.4778], [1.9717]]))
        self.bias2 = nn.Parameter(torch.tensor([1.2091, -0.3104]))

    def forward(self, x):
        if x.ndim < 2:
            x = torch.atleast_2d(x)
        x = x[:, :2]  # this net only reads age and edu_num
        offset = self.normalize_offset.to(x.device)
        scale = self.normalize_scale.to(x.device)
        x_norm = (x - offset) / scale
        h = F.linear(x_norm, self.weight1, self.bias1)
        o = F.linear(h, self.weight2, self.bias2)
        return o


class FairSquareNNV2H2(FairSquareV2BaseNN):
    """
    Since neuron neural network model from
    https://github.com/sedrews/fairsquare/blob/master/oopsla/noqual/M_ind_F_NN_V2_H2.fr
    """

    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(
            torch.tensor([[0.0326, 0.9998], [-0.4210, -0.6857]])
        )
        self.bias1 = nn.Parameter(torch.tensor([-0.8278, 0.8345]))
        self.weight2 = nn.Parameter(torch.tensor([[0.6004, 1.1573], [1.5905, 1.0555]]))
        self.bias2 = nn.Parameter(torch.tensor([0.8849, 0.9152]))

    def forward(self, x):
        if x.ndim < 2:
            x = torch.atleast_2d(x)
        x = x[:, :2]  # this net only reads age and edu_num
        x_norm = (x - self.normalize_offset) / self.normalize_scale
        h = F.linear(x_norm, self.weight1, self.bias1)
        o = F.linear(h, self.weight2, self.bias2)
        return o


class FairSquareNNV3H2(FairSquareV2BaseNN):
    """
    Since neuron neural network model from
    https://github.com/sedrews/fairsquare/blob/master/oopsla/noqual/M_ind_F_NN_V3_H2.fr
    """

    def __init__(self):
        super().__init__()
        # N_age = ((age - 17.0) / 73.0 - 0.5) * 10 + 0.5
        # N_education_num = ((education_num - 3.0) / 13.0 - 0.5) * 10 + 0.5
        # N_capital_gain = ((capital_gain - 0.0) / 22040.0 - 0.5) * 10 + 0.5
        self.normalize_offset = torch.tensor(
            [
                17.0 + 0.5 * 73 - 0.5 * 7.3,  # age
                3.0 + 0.5 * 13.0 - 0.5 * 1.3,  # education_num
                0.5 * 22040.0 - 0.5 * 2204.0,  # capital gain
            ]
        )
        self.normalize_scale = torch.tensor([7.3, 1.3, 2204.0])
        self.weight1 = nn.Parameter(
            torch.tensor([[-0.227, 0.6434, 2.3643], [-0.0236, -3.3556, -1.8183]])
        )
        self.bias1 = nn.Parameter(torch.tensor([3.7146, -1.7810]))
        self.weight2 = nn.Parameter(torch.tensor([[0.4865, 1.0685], [1.7044, -1.3880]]))
        self.bias2 = nn.Parameter(torch.tensor([-1.8079, 0.6830]))

    def forward(self, x):
        if x.ndim < 2:
            x = torch.atleast_2d(x)
        x = x[:, (0, 1, 3)]  # this net only reads age and edu_num and capital gain
        x_norm = (x - self.normalize_offset) / self.normalize_scale
        h = F.linear(x_norm, self.weight1, self.bias1)
        o = F.linear(h, self.weight2, self.bias2)
        return o
