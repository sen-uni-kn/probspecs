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
        "age": (0.0, 125.0),
        "education_num": (0.0, 125.0),
        "capital_gain": (0.0, 65000),
        "capital_loss": (0.0, 4000),
        "hours_per_week": (0.0, 150.0),
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


class _BayesianNetworkPopModel(nn.Module):
    def __init__(self):
        super().__init__()
        # there's some issue with auto_LiRPA and buffers
        self.continuous_locs = torch.tensor(
            [
                568.4105,  # capital gain 1 => sex < 1
                1329.3700,  # capital gen 2 => sex >= 1
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
            ]
        )
        self.continuous_scales = torch.tensor(
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
            ]
        )
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
                ]
            )
        )

    def forward(self, x):
        if x.ndim < 2:
            x = torch.atleast_2d(x)

        sex = x[:, 2]
        cont_values = F.linear(x, self.replicate_inputs)
        cont_locs = self.continuous_locs.to(device=x.device)
        cont_scales = self.continuous_scales.to(device=x.device)
        cont_values = cont_scales * cont_values + cont_locs

        const_zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        # centre bernoulli distribution at 0.0 to avoid the step at 0.0.
        male_indicator = torch.heaviside(sex - 0.5, values=const_zero)
        female_indicator = torch.heaviside(0.5 - sex, values=const_zero)

        capital_gain = (
            female_indicator * cont_values[:, 0] + male_indicator * cont_values[:, 1]
        )
        const_one = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        female_capital_gain_indicator = torch.heaviside(
            capital_gain - 7298.0, const_one
        )
        male_capital_gain_indicator = torch.heaviside(capital_gain - 5178.0, const_one)
        # these indicators are == 2 if the case matches
        case1_indicator = female_indicator + 1 - female_capital_gain_indicator
        case2_indicator = female_indicator + female_capital_gain_indicator
        case3_indicator = male_indicator + 1 - male_capital_gain_indicator
        case4_indicator = male_indicator + male_capital_gain_indicator
        case1_indicator = torch.heaviside(case1_indicator - 1.5, const_zero)
        case2_indicator = torch.heaviside(case2_indicator - 1.5, const_zero)
        case3_indicator = torch.heaviside(case3_indicator - 1.5, const_zero)
        case4_indicator = torch.heaviside(case4_indicator - 1.5, const_zero)

        age = (
            cont_values[:, 2] * case1_indicator
            + cont_values[:, 3] * case2_indicator
            + cont_values[:, 4] * case3_indicator
            + cont_values[:, 5] * case4_indicator
        )
        edu_num = (
            cont_values[:, 6] * case1_indicator
            + cont_values[:, 7] * case2_indicator
            + cont_values[:, 8] * case3_indicator
            + cont_values[:, 9] * case4_indicator
        )
        capital_loss = (
            cont_values[:, 10] * case1_indicator
            + cont_values[:, 11] * case2_indicator
            + cont_values[:, 12] * case3_indicator
            + cont_values[:, 13] * case4_indicator
        )
        hours_per_week = (
            cont_values[:, 14] * case1_indicator
            + cont_values[:, 15] * case2_indicator
            + cont_values[:, 16] * case3_indicator
            + cont_values[:, 17] * case4_indicator
        )
        return torch.hstack(
            [age, edu_num, sex, capital_gain, capital_loss, hours_per_week]
        )


class BayesianNetworkPopulationModel:
    """
    A bayesian network as a population model.
    """

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
        return _BayesianNetworkPopModel()
