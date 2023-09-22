# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from collections import OrderedDict
from enum import Enum, auto, unique
from typing import Sequence

import torch


class InputSpace:
    """
    A description of the input space of a neural network.
    An input space consists of several attributes.
    We consider continuous and one-hot encoded categorical
    attributes.
    Each attribute is equipped with a set of valid values.
    For continuous attributes this is an interval in real space, while
    for categorical attributes it's the set of valid values.

    For proving inputs to a neural network, the input attributes
    are encoded in a real-valued vector space.
    While the continuous attributes are passed on as-is,
    the categorical attributes are one-hot
    encoded, producing several dimensions in the vector space.
    """

    @unique
    class AttributeType(Enum):
        CONTINUOUS = auto()
        CATEGORICAL = auto()

    def __init__(
        self,
        attributes: Sequence[str],
        data_types: dict[str, AttributeType],
        continuous_ranges: dict[str, tuple[float, float]],
        categorical_values: dict[str, tuple[str, ...]],
    ):
        self.__attributes = tuple(
            (
                attr_name,
                data_types[attr_name],
                continuous_ranges[attr_name]
                if data_types[attr_name] is self.AttributeType.CONTINUOUS
                else categorical_values[attr_name],
            )
            for attr_name in attributes
        )

    @property
    def attribute_names(self) -> tuple[str, ...]:
        """
        The names of the attributes, ordered as the attributes are ordered.
        """
        return tuple(attr[0] for attr in self.__attributes)

    @property
    def attribute_types(self) -> tuple[str, ...]:
        """
        The types (continuous/categorical) of the attributes,
        ordered as the attributes are ordered.
        """
        return tuple(attr[0] for attr in self.__attributes)

    def attribute_name(self, index: int) -> str:
        """
        The name of the i-th attribute.

        :param index: The index of the attribute (i).
        """
        return self.__attributes[index][0]

    def attribute_type(self, index: int) -> AttributeType:
        """
        The type of the i-th attribute.

        :param index: The index of the attribute (i).
        """
        return self.__attributes[index][1]

    def attribute_bounds(self, index: int) -> tuple[float, float]:
        """
        The input bounds of the i-th attribute (continuous).

        :param index: The index of the attribute (i).
        :raises ValueError: If the i-th attribute isn't continuous.
        """
        attr_name, attr_type, attr_info = self.__attributes[index]
        if attr_type is not self.AttributeType.CONTINOUS:
            raise ValueError(f"Attribute {attr_name} has no input bounds.")
        return attr_info

    def attribute_values(self, index: int) -> tuple[str, ...]:
        """
        The permitted values of the i-th attribute (categorical).

        :param index: The index of the attribute (i).
        :raises ValueError: If the i-th attribute isn't categorical.
        """
        attr_name, attr_type, attr_info = self.__attributes[index]
        if attr_type is not self.AttributeType.CATEGORICAL:
            raise ValueError(f"Attribute {attr_name} has no input values.")
        return attr_info

    @property
    def encoding_shape(self) -> int:
        """
        The shape of the encoding space.
        """
        return sum(
            len(attr_info) if attr_type is self.AttributeType.CATEGORICAL else 1
            for _, attr_type, attr_info in self.__attributes
        )

    @property
    def encoding_layout(self) -> OrderedDict[str, tuple[int]]:
        """
        The layout of the real-valued vector space that the input space is
        encoded in. This layout is a mapping from attributes to the dimensions
        of the vector space into which the attributes are mapped.
        While continuous attributes occupy a single dimension in the vector space,
        a categorical attribute occupies as many dimensions as the attribute has values.
        """
        layout = OrderedDict()
        i = 0
        for attr_name, attr_type, attr_info in self.__attributes:
            match attr_type:
                case self.AttributeType.CONTINUOUS:
                    layout[attr_name] = (i,)
                case self.AttributeType.CATEGORICAL:
                    layout[attr_name] = tuple(range(i, i + len(attr_info)))
                case _:
                    raise NotImplementedError()
            i += len(layout[attr_name])
        return layout

    @property
    def encoding_domain(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The hyperrectangular domain of the encoding space.
        This contains the upper and lower bounds of all continuous variables,
        as well as zeros and ones for the dimensions into which categorical
        variables are mapped.
        """
        lbs = []
        ubs = []
        for attr_name, attr_type, attr_info in self.__attributes:
            match attr_type:
                case self.AttributeType.CONTINUOUS:
                    lbs.append(attr_info[0])
                    ubs.append(attr_info[1])
                case self.AttributeType.CATEGORICAL:
                    lbs.append([0] * len(attr_info))
                    ubs.append([1] * len(attr_info))
                case _:
                    raise NotImplementedError()
        return torch.tensor(lbs), torch.tensor(ubs)

    def encode(self, x: Sequence[float | str]) -> torch.Tensor:
        """
        Encode an input :code:`x` into the real-valued vector encoding space.

        :param x: The input to encode. A sequence of values for the continuous
         and categorical variables in the order of the attributes.
        :return: The encoded input.
        """
        encoding = []
        for value, (attr_name, attr_type, attr_info) in zip(
            x, self.__attributes, strict=True
        ):
            match attr_type:
                case self.AttributeType.CONTINUOUS:
                    if not isinstance(value, float):
                        raise ValueError(
                            f"Invalid value for continuous attribute {attr_name}: {value}"
                        )
                    lb, ub = attr_info
                    if not lb <= value <= ub:
                        raise ValueError(
                            f"Invalid value for continuous attribute {attr_name} "
                            f"with bounds [{lb}, {ub}]: {value}"
                        )
                    encoding.append(value)
                case self.AttributeType.CATEGORICAL:
                    if not isinstance(value, str):
                        raise ValueError(
                            f"Invalid value for categorical attribute {attr_name}: "
                            f"{value}"
                        )
                    if value not in attr_info:
                        raise ValueError(
                            f"Invalid value for categorical attribute {attr_name} "
                            f"with values {attr_info}: {value}"
                        )
                    for category in attr_info:
                        encoding.append(1 if category == value else 0)
                case _:
                    raise NotImplementedError()
        return torch.tensor(encoding)

    def decode(self, x: torch.Tensor) -> tuple[float | str, ...]:
        """
        Decode an input :code:`x` from it's real-value vector encoding.

        :param x: The input to decode.
        :return: The input as a sequence of values for the continuous
         and categorical variables in the order of the attributes.
        """
        if x.ndim != 1 or x.size(0) != self.encoding_shape:
            raise ValueError(
                f"Not an encoding of this input space: {x} (Dimension mismatch)"
            )
        decoding = []
        layout = self.encoding_layout
        for attr_name, attr_type, attr_info in self.__attributes:
            match attr_type:
                case self.AttributeType.CONTINUOUS:
                    value = x[layout[attr_name][0]]
                    lb, ub = attr_info
                    if not lb <= value <= ub:
                        raise ValueError(
                            f"Invalid value for continuous attribute {attr_name} "
                            f"with bounds [{lb}, {ub}]: {value}"
                        )
                case self.AttributeType.CATEGORICAL:
                    one_hot = x.index_select(0, torch.tensor(layout[attr_name]))
                    value = None
                    for i, val in enumerate(one_hot):
                        if not torch.isclose(
                            val, torch.zeros(())
                        ) and not torch.isclose(val, torch.ones(())):
                            raise ValueError(
                                f"Invalid one-hot encoding of categorical attribute {attr_name}: {one_hot}"
                            )
                        if torch.isclose(val, torch.ones(())):
                            if value is not None:
                                raise ValueError(
                                    f"Invalid one-hot encoding of categorical attribute {attr_name}: {one_hot}"
                                )
                            value = attr_info[i]
                case _:
                    raise NotImplementedError()
            decoding.append(value)
        return tuple(decoding)
