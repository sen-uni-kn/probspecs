# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from probspecs import TabularInputSpace
from adult import Adult

_CONT = TabularInputSpace.AttributeType.CONTINUOUS
_ORD = TabularInputSpace.AttributeType.ORDINAL
_CAT = TabularInputSpace.AttributeType.CATEGORICAL

adult_input_space = TabularInputSpace(
    attributes=tuple(Adult.variables.keys()),
    data_types={
        var: _CONT if values is None else _CAT
        for var, values in Adult.variables.items()
    },
    continuous_ranges={
        "age": (17.0, 90.0),
        "fnlwgt": (10000.0, 1500000.0),
        "education-num": (1.0, 16.0),
        "capital-gain": (0.0, 99999.0),
        "capital-loss": (0.0, 5000.0),
        "hours-per-week": (1.0, 99.0),
    },
    ordinal_ranges={},
    categorical_values={
        var: values for var, values in Adult.variables.items() if values is not None
    },
)
