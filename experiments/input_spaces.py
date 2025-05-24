#  Copyright (c) 2023-2024. David Boetius
#  Licensed under the MIT License
from torchstats import TabularInputSpace
from fairnessdatasets import Adult, SouthGerman

_INT = TabularInputSpace.AttributeType.INTEGER
_CAT = TabularInputSpace.AttributeType.CATEGORICAL

adult_input_space = TabularInputSpace(
    attributes=tuple(Adult.variables.keys()),
    data_types={
        var: _INT if values is None else _CAT for var, values in Adult.variables.items()
    },
    continuous_ranges={},
    integer_ranges={
        "age": (17, 90),
        "fnlwgt": (10000, 1500000),
        "education-num": (1, 16),
        "capital-gain": (0, 999990),
        "capital-loss": (0, 5000),
        "hours-per-week": (1, 99),
    },
    categorical_values={
        var: values for var, values in Adult.variables.items() if values is not None
    },
)
"""
An input space describing the Adult dataset.

When unnormalized, all non-categorical variables are integer variables
in the Adult dataset.
"""

south_german_input_space = TabularInputSpace(
    attributes=tuple(SouthGerman.variables.keys()),
    data_types={
        var: _INT if values is None else _CAT
        for var, values in SouthGerman.variables.items()
    },
    continuous_ranges={},
    integer_ranges={"duration": (4, 72), "age": (19, 75), "amount": (250, 20_000)},
    categorical_values={
        var: tuple(values.keys())
        for var, values in SouthGerman.variables.items()
        if values is not None
    },
)
