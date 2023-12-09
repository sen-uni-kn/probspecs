# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
from torch import nn
import torch.nn.functional as F


class FairSquareNNV2H1(nn.Module):
    """
    Since neuron neural network model from
    https://github.com/sedrews/fairsquare/blob/master/oopsla/noqual/M_ind_F_NN_V2_H1.fr
    """

    def __init__(self):
        super().__init__()
        self.normalize_offset = torch.tensor(
            [
                10 * 17.0 / 73.0 + 10 * 0.5 - 0.5,  # age
                10.0 * 3.0 / 13.0 + 10.0 * 0.5 - 0.5,  # education_num
            ]
        )
        self.normalize_scale = torch.tensor([73.0 / 10.0, 13.0 / 10.0])
        self.weight1 = nn.Parameter(torch.tensor([[0.1718, 1.1416]]))
        self.bias1 = nn.Parameter(torch.tensor([0.4754]))
        self.weight2 = nn.Parameter(torch.tensor([[0.4778], [1.9717]]))
        self.bias2 = nn.Parameter(torch.tensor([1.2091, -0.3104]))

    def forward(self, x):
        if x.ndim < 2:
            x = torch.atleast_2d(x)
        x = x[:, :2]  # this net only reads age and edu_num
        x_norm = x / self.normalize_scale - self.normalize_offset
        h = F.linear(x_norm, self.weight1, self.bias1)
        o = F.linear(h, self.weight2, self.bias2)
        return o
