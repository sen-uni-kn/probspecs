# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
# Based on the FairSquare repository: https://github.com/sedrews/fairsquare
"""
Recreates the population models from the FairSquare repository:
https://github.com/sedrews/fairsquare.

FairSquare does not use (or require) bounds on the values of the random variables.
Since we need such bounds, we pick sensible, but conservative, ranges based on the
meaning of a variable.
For example, we pick a range of 0 to 125 for the variable "age".
We enforce these bounds also in the bayesian network population models.

We assume that the education_num variable measures the years of education,
based on tbe integrity constraint in https://github.com/sedrews/fairsquare/blob/bd27437a8a93ec4a239ae99edd74c69c46e9ee4b/oopsla/noqual/M_BNc_F_NN_V2_H1.fr
"""
from math import sqrt
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import norm, bernoulli

from probspecs import (
    TabularInputSpace,
    ContinuousDistribution1d,
    MultidimensionalIndependent,
    ProbabilityDistribution,
    InputSpace,
)
from probspecs.probability_distribution import DiscreteDistribution1d

AttrT = TabularInputSpace.AttributeType


_input_lbs = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_input_ubs = (125.0, 125.0, 1.0, 65000.0, 4000.0, 150.0)

_classifier_input_space = TabularInputSpace(
    attributes=(
        "age",
        "education_num",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ),
    data_types={
        "age": AttrT.CONTINUOUS,
        "education_num": AttrT.CONTINUOUS,
        "sex": AttrT.ORDINAL,
        "capital_gain": AttrT.CONTINUOUS,
        "capital_loss": AttrT.CONTINUOUS,
        "hours_per_week": AttrT.CONTINUOUS,
    },
    continuous_ranges={
        "age": (_input_lbs[0], _input_ubs[0]),
        "education_num": (_input_lbs[1], _input_ubs[1]),
        "capital_gain": (_input_lbs[3], _input_ubs[3]),
        "capital_loss": (_input_lbs[4], _input_ubs[4]),
        "hours_per_week": (_input_lbs[5], _input_ubs[5]),
    },
    ordinal_ranges={"sex": (0, 1)},
    categorical_values={},
)


class IndependentPopulationModel:
    """
    A population model composed from independent variables.
    """

    @property
    def input_space(self) -> InputSpace:
        return _classifier_input_space

    @property
    def probability_distribution(self) -> ProbabilityDistribution:
        age_distr = ContinuousDistribution1d(norm(loc=38.5816, scale=sqrt(186.0614)))
        edu_num_distr = ContinuousDistribution1d(norm(loc=10.0806, scale=sqrt(6.6188)))
        sex_distr = DiscreteDistribution1d(
            bernoulli(0.6693)
        )  # 67% males in the dataset
        capital_gain = ContinuousDistribution1d(
            norm(loc=1077.6488, scale=sqrt(54542539.1784))
        )
        capital_loss = ContinuousDistribution1d(
            norm(loc=87.3038, scale=sqrt(162376.9378))
        )
        hours_per_week = ContinuousDistribution1d(
            norm(loc=40.4374, scale=sqrt(152.4589))
        )
        return MultidimensionalIndependent(
            age_distr,
            edu_num_distr,
            sex_distr,
            capital_gain,
            capital_loss,
            hours_per_week,
            input_shape=(6,),
        )

    @property
    def population_model(self) -> Optional[torch.nn.Module]:
        return None


# Input space for base normal distributions with mean 0.0 and 1.0.
# These distributions are rescaled by the population model.
_base_input_space = TabularInputSpace(
    attributes=(
        "age",
        "education_num",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ),
    data_types={
        "age": AttrT.CONTINUOUS,
        "education_num": AttrT.CONTINUOUS,
        "sex": AttrT.ORDINAL,
        "capital_gain": AttrT.CONTINUOUS,
        "capital_loss": AttrT.CONTINUOUS,
        "hours_per_week": AttrT.CONTINUOUS,
    },
    continuous_ranges={
        "age": (-10.0, 10.0),
        "education_num": (-10.0, 10.0),
        "capital_gain": (-10.0, 10.0),
        "capital_loss": (-10.0, 10.0),
        "hours_per_week": (-10.0, 10.0),
    },
    ordinal_ranges={"sex": (0, 1)},
    categorical_values={},
)


class _BayesianNetworkPopModelTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.continuous_locs = torch.tensor(
            [
                38.4208,  # age 1
                38.8125,  # age 2
                10.0827,  # education num 1
                10.1041,  # education num 2
                0.0,  # sex
            ]
        )
        self.continuous_scales = torch.tensor(
            [
                sqrt(184.9151),  # age 1
                sqrt(193.4918),  # age 2
                sqrt(6.5096),  # education num 1
                sqrt(6.1522),  # education num 2
                1.0,  # sex
            ]
        )
        self.replicate_inputs = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # age
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # edu num
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # sex
                ]
            )
        )
        self.replicate_sex = nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # age 1
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # age 2
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # edu num 1
                    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # edu num 2
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # for sex in cont_values
                ]
            )
        )
        self.threshold_sex = nn.Parameter(torch.tensor([0.5, -0.5, 0.5, -0.5, 0.0]))
        self.reduce_output = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        self.offset_output = nn.Parameter(
            torch.tensor([0.0, 0.0, 0.0, 1329.3700, 86.5949, 40.3897])
        )

    def forward(self, x):
        if x.ndim < 2:
            x = torch.atleast_2d(x)

        cont_values = F.linear(x, self.replicate_inputs)
        cont_locs = self.continuous_locs.to(device=x.device)
        cont_scales = self.continuous_scales.to(device=x.device)
        cont_values = cont_scales * cont_values + cont_locs

        sex = F.linear(x, self.replicate_sex, self.threshold_sex)
        const_zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        sex_indicator = torch.heaviside(sex, values=const_zero)

        cont_values = cont_values * sex_indicator
        return F.linear(cont_values, self.reduce_output, self.offset_output)


class _BayesianNetworkPopModel(nn.Module):
    def __init__(self, integrity_constraint: bool = False, clip_outputs: bool = False):
        super().__init__()
        self.integrity_constraint = integrity_constraint
        self.clip_outputs = clip_outputs
        # there's some issue with auto_LiRPA and buffers,
        # so register these tensors as parameters
        self.continuous_locs = nn.Parameter(
            torch.tensor(
                [
                    568.4105,  # capital gain 1 <- sex < 1
                    1329.3700,  # capital gen 2 <- sex >= 1
                    38.4208,  # age 1
                    38.8125,  # age 2
                    38.6361,  # age 3
                    38.2668,  # age 4
                    10.0827,  # education num 1
                    10.1041,  # education num 2
                    10.0817,  # education num 3
                    10.0974,  # education num 4
                    86.5949,  # capital loss 1
                    117.8083,  # capital loss 2
                    87.0152,  # capital loss 3
                    101.7672,  # capital loss 4
                    40.4959,  # hours per week 1
                    41.6916,  # hours per week 2
                    40.3897,  # hours per week 3
                    40.6473,  # hours per week 4
                    0.0,  # sex
                ]
            )
        )
        self.continuous_scales = nn.Parameter(
            torch.tensor(
                [
                    sqrt(24248365.5428),  # capital gain 1
                    sqrt(69327473.1006),  # capital gain 2
                    sqrt(184.9151),  # age 1
                    sqrt(193.4918),  # age 2
                    sqrt(187.2435),  # age 3
                    sqrt(187.2747),  # age 4
                    sqrt(6.5096),  # education num 1
                    sqrt(6.1522),  # education num 2
                    sqrt(6.4841),  # education num 3
                    sqrt(7.1793),  # education num 4
                    sqrt(157731.9553),  # capital loss 1
                    sqrt(252612.0300),  # capital loss 2
                    sqrt(161032.4157),  # capital loss 3
                    sqrt(189798.1926),  # capital loss 4
                    sqrt(151.4148),  # hours per week 1
                    sqrt(165.3773),  # hours per week 2
                    sqrt(150.6723),  # hours per week 3
                    sqrt(153.4823),  # hours per week 4
                    1.0,  # sex
                ]
            )
        )
        # create several copies of each input using a linear layer
        self.replicate_inputs = nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # capital gain
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # age
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # education num
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # capital loss
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # working hours
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # sex
                ]
            )
        )
        self.process_sex_weight = nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        self.process_sex_bias = nn.Parameter(torch.tensor([0.5, -0.5]))
        self.capital_gain_thresholds = nn.Parameter(torch.tensor([7298.0, 5178.0]))
        self.sex_to_cases = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            )
        )
        self.capital_gain_to_cases = nn.Parameter(
            torch.tensor(
                [
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                ]
            )
        )
        self.capital_gain_to_cases_bias = nn.Parameter(
            torch.tensor([1.0, 0.0, 1.0, 0.0])
        )
        self.replicate_case_indicator = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0],  # capital gain: case1 + case2 = female?
                    [0.0, 0.0, 1.0, 1.0],  # case3 + case4 = male?
                    [1.0, 0.0, 0.0, 0.0],  # age
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],  # education num
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],  # capital loss
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],  # working hours
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],  # sex
                ]
            )
        )
        self.reduce_values = nn.Parameter(
            torch.tensor(
                [
                    [  # age
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [  # education num
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [  # sex
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    [  # capital gain
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [  # capital loss
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [  # working hours
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                    ],
                ]
            )
        )
        self.extract_age_and_edu_num = nn.Parameter(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
            )
        )
        self.clear_edu_num = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        self.enlarge_edu_num = nn.Parameter(
            torch.tensor([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
        )
        self.output_mins = nn.Parameter(torch.tensor(_input_lbs))
        self.output_maxs = nn.Parameter(torch.tensor(_input_ubs))
        self.const_one = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if x.ndim < 2:
            x = torch.atleast_2d(x)

        y = F.linear(x, self.replicate_inputs)
        y = self.continuous_scales * y + self.continuous_locs

        sex_values = F.linear(x, self.process_sex_weight, self.process_sex_bias)
        sex_indicator = torch.heaviside(sex_values, self.const_one)

        capital_gain = sex_indicator * y[:, :2]
        capital_gain = capital_gain - self.capital_gain_thresholds
        capital_gain_indicator = torch.heaviside(capital_gain, self.const_one)

        # these indicators are == 2 if the case matches
        # case1 = female? + 1 - capital_gain_female
        # case2 = female? + capital_gain_female
        # case3 = male? + 1 - capital_gain_male
        # case4 = male? + capital_gain_male
        sex_indicator = F.linear(sex_indicator, self.sex_to_cases)
        capital_gain_indicator = F.linear(
            capital_gain_indicator,
            self.capital_gain_to_cases,
            self.capital_gain_to_cases_bias,
        )
        case_indicator = sex_indicator + capital_gain_indicator
        case_indicator = torch.heaviside(case_indicator - 1.5, self.const_one)
        case_indicator = F.linear(case_indicator, self.replicate_case_indicator)

        y = y * case_indicator
        z = F.linear(y, self.reduce_values)

        if self.integrity_constraint:
            age_and_edu_num = F.linear(z, self.extract_age_and_edu_num)
            edu_num = torch.minimum(age_and_edu_num[:, 1:], age_and_edu_num[:, :1])
            z = F.linear(z, self.clear_edu_num) + F.linear(
                edu_num, self.enlarge_edu_num
            )

        if self.clip_outputs:
            z = torch.clamp(z, min=self.output_mins, max=self.output_maxs)
        return z


class BayesianNetworkPopulationModel:
    """
    A bayesian network as a population model.
    """

    def __init__(self, integrity_constraint: bool = False, clip_outputs: bool = False):
        self.__pop_model = _BayesianNetworkPopModel(integrity_constraint, clip_outputs)

    @property
    def input_space(self) -> InputSpace:
        return _base_input_space

    @property
    def probability_distribution(self) -> ProbabilityDistribution:
        # 67% males in the dataset
        sex_distr = DiscreteDistribution1d(bernoulli(0.6693))
        # distributions are rescaled by the population model.
        age_distr = ContinuousDistribution1d(norm(loc=0.0, scale=1.0))
        edu_num_distr = ContinuousDistribution1d(norm(loc=0.0, scale=1.0))
        capital_gain = ContinuousDistribution1d(norm(loc=0.0, scale=1.0))
        capital_loss = ContinuousDistribution1d(norm(loc=0.0, scale=1.0))
        hours_per_week = ContinuousDistribution1d(norm(loc=0.0, scale=1.0))
        return MultidimensionalIndependent(
            age_distr,
            edu_num_distr,
            sex_distr,
            capital_gain,
            capital_loss,
            hours_per_week,
            input_shape=(6,),
        )

    @property
    def population_model(self) -> Optional[torch.nn.Module]:
        return self.__pop_model
