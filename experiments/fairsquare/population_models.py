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

import torch.nn
from scipy.stats import norm, bernoulli

from probspecs import (
    TabularInputSpace,
    Distribution1d,
    MultidimensionalIndependent,
    ProbabilityDistribution,
    InputSpace,
)

AttrT = TabularInputSpace.AttributeType


_input_space = TabularInputSpace(
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
        return _input_space

    @property
    def probability_distribution(self) -> ProbabilityDistribution:
        age_distr = Distribution1d(norm(loc=38.5816, scale=sqrt(186.0614)))
        edu_num_distr = Distribution1d(norm(loc=10.0806, scale=sqrt(6.6188)))
        sex_distr = Distribution1d(bernoulli(0.6693))  # 67% males in the dataset
        capital_gain = Distribution1d(norm(loc=1077.6488, scale=sqrt(54542539.1784)))
        capital_loss = Distribution1d(norm(loc=87.3038, scale=sqrt(162376.9378)))
        hours_per_week = Distribution1d(norm(loc=40.4374, scale=sqrt(152.4589)))
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
