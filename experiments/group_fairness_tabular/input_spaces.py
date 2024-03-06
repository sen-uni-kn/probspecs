# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from probspecs import TabularInputSpace
from adult import Adult

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
