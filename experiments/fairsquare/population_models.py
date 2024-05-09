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
from scipy.stats import norm, truncnorm

from probspecs import TabularInputSpace, InputSpace
from probspecs.distributions import (
    ProbabilityDistribution,
    MultivariateIndependent,
    Categorical,
    BayesianNetwork,
)
from probspecs import distributions
import probspecs.operations as ops

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
        "sex": AttrT.INTEGER,
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
    integer_ranges={"sex": (0, 1)},
    categorical_values={},
)


# Here, we choose the lower and upper bounds such that the cdf of all
# distributions is 0.0/1.0 to machine precision
_unrealistic_classifier_input_space = TabularInputSpace(
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
        "sex": AttrT.INTEGER,
        "capital_gain": AttrT.CONTINUOUS,
        "capital_loss": AttrT.CONTINUOUS,
        "hours_per_week": AttrT.CONTINUOUS,
    },
    continuous_ranges={
        "age": (-500.0, 200.0),
        "education_num": (-100.0, 40.0),
        "capital_gain": (-1000000.0, 10000.0),
        "capital_loss": (-20000.0, 5000.0),
        "hours_per_week": (-500.0, 200.0),
    },
    integer_ranges={"sex": (0, 1)},
    categorical_values={},
)


class IndependentPopulationModel:
    """
    A population model composed from independent variables.
    """

    def __init__(self, realistic=True):
        self.realistic = realistic

    @property
    def input_space(self) -> InputSpace:
        return (
            _classifier_input_space
            if self.realistic
            else _unrealistic_classifier_input_space
        )

    @property
    def probability_distribution(self) -> ProbabilityDistribution:
        age_distr = distributions.wrap(norm(loc=38.5816, scale=sqrt(186.0614)))
        edu_num_distr = distributions.wrap(norm(loc=10.0806, scale=sqrt(6.6188)))
        sex_distr = Categorical([0.3307, 0.6693])
        capital_gain = distributions.wrap(
            norm(loc=1077.6488, scale=sqrt(54542539.1784))
        )
        capital_loss = distributions.wrap(norm(loc=87.3038, scale=sqrt(162376.9378)))
        hours_per_week = distributions.wrap(norm(loc=40.4374, scale=sqrt(152.4589)))
        return MultivariateIndependent(
            age_distr,
            edu_num_distr,
            sex_distr,
            capital_gain,
            capital_loss,
            hours_per_week,
            event_shape=(6,),
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
        "sex": AttrT.INTEGER,
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
    integer_ranges={"sex": (0, 1)},
    categorical_values={},
)


class _BayesianNetworkPopModel(nn.Module):
    def __init__(self):
        super().__init__()
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
        return z


class _IntegrityConstraint(nn.Module):
    def __init__(self, integrity_constraint: bool = False, clip_outputs: bool = False):
        super().__init__()
        self.integrity_constraint = integrity_constraint
        self.clip_outputs = clip_outputs

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

    def forward(self, x):
        if x.ndim < 2:
            x = torch.atleast_2d(x)

        if self.integrity_constraint:
            age_and_edu_num = F.linear(x, self.extract_age_and_edu_num)
            edu_num = torch.minimum(age_and_edu_num[:, 1:], age_and_edu_num[:, :1])
            x = F.linear(x, self.clear_edu_num) + F.linear(
                edu_num, self.enlarge_edu_num
            )

        if self.clip_outputs:
            x = ops.clamp(x, min=self.output_mins, max=self.output_maxs)
        return x


class BayesianNetworkPopulationModel:
    """
    A bayesian network as a population model.
    """

    def __init__(self, integrity_constraint: bool = False, clip_outputs: bool = False):
        self.__pop_model = nn.Sequential(
            _BayesianNetworkPopModel(),
            _IntegrityConstraint(integrity_constraint, clip_outputs),
        )

    @property
    def input_space(self) -> InputSpace:
        return _base_input_space

    @property
    def probability_distribution(self) -> ProbabilityDistribution:
        # 67% males in the dataset
        sex_distr = Categorical([0.3307, 0.6693])
        # distributions are rescaled by the population model.
        age_distr = distributions.wrap(norm(loc=0.0, scale=1.0))
        edu_num_distr = distributions.wrap(norm(loc=0.0, scale=1.0))
        capital_gain = distributions.wrap(norm(loc=0.0, scale=1.0))
        capital_loss = distributions.wrap(norm(loc=0.0, scale=1.0))
        hours_per_week = distributions.wrap(norm(loc=0.0, scale=1.0))
        return MultivariateIndependent(
            age_distr,
            edu_num_distr,
            sex_distr,
            capital_gain,
            capital_loss,
            hours_per_week,
            event_shape=(6,),
        )

    @property
    def population_model(self) -> Optional[torch.nn.Module]:
        return self.__pop_model


def _get_explicit_bayesian_network(realistic: bool = True):
    if realistic:
        in_space = _classifier_input_space
    else:
        in_space = _unrealistic_classifier_input_space

    bn_factory = BayesianNetwork.Factory()
    sex = bn_factory.new_node("sex")
    sex.discrete_event_space([0.0], [1.0])
    sex.set_conditional_probability({}, Categorical([0.3307, 0.6693]))

    capital_gain = bn_factory.new_node("capital_gain")
    capital_gain.add_parent(sex)
    capital_gain_min, capital_gain_max = in_space.attribute_bounds("capital_gain")
    capital_gain.continuous_event_space(capital_gain_min, capital_gain_max)
    # sex=0
    loc = 568.4105
    scale = sqrt(24248365.5428)
    a, b = (capital_gain_min - loc) / scale, (capital_gain_max - loc) / scale
    capital_gain.set_conditional_probability(
        {sex: [0.0]}, distributions.wrap(truncnorm(a, b, loc=loc, scale=scale))
    )
    # sex=1
    loc = 1329.3700
    scale = sqrt(69327473.1006)
    a, b = (capital_gain_min - loc) / scale, (capital_gain_max - loc) / scale
    capital_gain.set_conditional_probability(
        {sex: [1.0]}, distributions.wrap(truncnorm(a, b, loc=loc, scale=scale))
    )

    def make_node(var, case1, case2, case3, case4):
        node = bn_factory.new_node(var)
        node.set_parents(sex, capital_gain)
        min_, max_ = in_space.attribute_bounds(var)
        node.continuous_event_space(min_, max_)

        # case1: sex=0, capital_gain < 7298.0
        step_female = 7298.0
        loc, scale = case1
        a, b = (min_ - loc) / scale, (max_ - loc) / scale
        node.set_conditional_probability(
            {sex: [0.0], capital_gain: ([capital_gain_min], [step_female])},
            distributions.wrap(truncnorm(a, b, loc=loc, scale=scale)),
        )
        # case2: sex=0, capital_gain >= 7298.0
        loc, scale = case2
        a, b = (min_ - loc) / scale, (max_ - loc) / scale
        node.set_conditional_probability(
            {sex: [0.0], capital_gain: ([step_female], [capital_gain_max])},
            distributions.wrap(truncnorm(a, b, loc=loc, scale=scale)),
        )
        # case3: sex=1, capital_gain < 5178.0
        step_male = 5178.0
        loc, scale = case3
        a, b = (min_ - loc) / scale, (max_ - loc) / scale
        node.set_conditional_probability(
            {sex: [1.0], capital_gain: ([capital_gain_min], [step_male])},
            distributions.wrap(truncnorm(a, b, loc=loc, scale=scale)),
        )
        # case4: sex=1, capital_gain >= 5178.0
        loc, scale = case4
        a, b = (min_ - loc) / scale, (max_ - loc) / scale
        node.set_conditional_probability(
            {sex: [1.0], capital_gain: ([step_male], [capital_gain_max])},
            distributions.wrap(truncnorm(a, b, loc=loc, scale=scale)),
        )

    make_node(
        "age",
        (38.4208, sqrt(184.9151)),
        (38.8125, sqrt(193.4918)),
        (38.6361, sqrt(187.2435)),
        (38.2668, sqrt(187.2435)),
    )
    make_node(
        "education_num",
        (10.0827, sqrt(6.5096)),
        (10.1041, sqrt(6.1522)),
        (10.0817, sqrt(6.4841)),
        (10.0974, sqrt(7.1793)),
    )
    make_node(
        "capital_loss",
        (86.5949, sqrt(157731.9553)),
        (117.8083, sqrt(252612.0300)),
        (87.0152, sqrt(161032.4157)),
        (101.7672, sqrt(189798.1926)),
    )
    make_node(
        "hours_per_week",
        (40.4959, sqrt(151.4148)),
        (41.6916, sqrt(165.3773)),
        (40.3897, sqrt(150.6723)),
        (40.6473, sqrt(153.4823)),
    )
    bn_factory.reorder_nodes(_classifier_input_space.attribute_names)
    return bn_factory.create()


_explicit_bayesian_network = _get_explicit_bayesian_network(realistic=False)
_realistic_bayesian_network = _get_explicit_bayesian_network()


class ExplicitBayesianNetworkPopulationModel:
    def __init__(
        self,
        realistic: bool = True,
        integrity_constraint: bool = False,
        clip_outputs: bool = False,
    ):
        """
        :param realistic: Use classifier input bounds.
        """
        self.realistic = realistic
        self.__transform = _IntegrityConstraint(integrity_constraint, clip_outputs)

    @property
    def input_space(self) -> InputSpace:
        return (
            _classifier_input_space
            if self.realistic
            else _unrealistic_classifier_input_space
        )

    @property
    def probability_distribution(self) -> ProbabilityDistribution:
        return (
            _realistic_bayesian_network
            if self.realistic
            else _explicit_bayesian_network
        )

    @property
    def population_model(self) -> Optional[torch.nn.Module]:
        return self.__transform
