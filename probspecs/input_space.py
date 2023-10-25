# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from collections import OrderedDict
from enum import Enum, auto, unique
from math import prod
from typing import Sequence, Protocol

import torch


class InputSpace(Protocol):
    """
    A description of an input space of a neural network
    """

    @property
    def input_shape(self) -> torch.Size:
        """
        The shape of the tensor supplied to a neural network.
        """
        raise NotImplementedError()

    @property
    def input_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The lower and upper edges of the hyperrectangular input domain
        that this object describes.
        """
        raise NotImplementedError()


class TensorInputSpace(InputSpace):
    """
    A regular real-valued tensor-shaped input space.
    """

    def __init__(self, lbs: torch.Tensor, ubs: torch.Tensor):
        self.lbs = lbs
        self.ubs = ubs

    @property
    def input_shape(self) -> torch.Size:
        return self.lbs.shape

    @property
    def input_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lbs, self.ubs


class TabularInputSpace(InputSpace):
    """
    A description of a tabular input space.
    Such an input space consists of several attributes.
    We consider continuous and one-hot encoded categorical
    attributes.
    Each attribute is equipped with a set of valid values.
    For continuous attributes this is an interval in real space, while
    for categorical attributes it's the set of valid values.

    For proving inputs to a neural network, the attributes
    are encoded in a real-valued vector space.
    While the continuous attributes are passed on as-is,
    the categorical attributes are one-hot
    encoded, producing several dimensions in the vector space.
    We call the real-valued vector space the
    *encoding space*, but also the actual *input space*.
    Contrary to this, the continuous and categorical attributes before encoding
    reside in the *attribute space*.
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
    def attribute_types(self) -> tuple[AttributeType, ...]:
        """
        The types (continuous/categorical) of the attributes,
        ordered as the attributes are ordered.
        """
        return tuple(attr[1] for attr in self.__attributes)

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
        if attr_type is not self.AttributeType.CONTINUOUS:
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
    def input_shape(self) -> torch.Size:
        """
        The shape of the encoding space.
        The encoding space is one-dimensional.
        """
        input_size = sum(
            len(attr_info) if attr_type is self.AttributeType.CATEGORICAL else 1
            for _, attr_type, attr_info in self.__attributes
        )
        return torch.Size([input_size])

    @property
    def input_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The hyper-rectangular domain of the encoding space.
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
                    lbs.extend([0.0] * len(attr_info))
                    ubs.extend([1.0] * len(attr_info))
                case _:
                    raise NotImplementedError()
        return torch.tensor(lbs), torch.tensor(ubs)

    @property
    def encoding_layout(self) -> OrderedDict[str, int | OrderedDict[str, int]]:
        """
        The layout of the real-valued vector space that the input space is
        encoded in. This layout is a mapping from attributes to the dimensions
        of the vector space into which the attributes are mapped.
        While continuous attributes occupy a single dimension in the vector space,
        a categorical attribute occupies as many dimensions as the attribute has values.
        For categorical attributes, the dimensions to which the attribute is mapped
        is a mapping from attribute values to dimensions.
        The dimension for a value is the dimension indicating whether the value
        is taken on in the one-hot encoding of the categorical attribute.

        Example:
         - Attributes: age: continuous, color: categorical (blue, red, green, other)
         - Encoding layout:
           `{age: 0, color: {blue: 1, red: 2, green: 3, other: 4}}`
        """
        layout: OrderedDict[str, int | OrderedDict] = OrderedDict()
        i = 0
        for attr_name, attr_type, attr_info in self.__attributes:
            match attr_type:
                case self.AttributeType.CONTINUOUS:
                    layout[attr_name] = i
                    i += 1
                case self.AttributeType.CATEGORICAL:
                    layout[attr_name] = OrderedDict(
                        zip(attr_info, range(i, i + len(attr_info)), strict=True)
                    )
                    i += len(attr_info)
                case _:
                    raise NotImplementedError()
        return layout

    def encode(self, x: Sequence[float | str]) -> torch.Tensor:
        """
        Encode a set of attributes :code:`x` into the real-valued encoding/input space,
        such that they can be fed to a neural network.

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
                        encoding.append(1.0 if category == value else 0.0)
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
        if x.ndim != 1 or x.size(0) != self.input_shape[0]:
            raise ValueError(
                f"Not an encoding of this input space: {x} (Dimension mismatch)"
            )
        decoding = []
        layout = self.encoding_layout
        for attr_name, attr_type, attr_info in self.__attributes:
            match attr_type:
                case self.AttributeType.CONTINUOUS:
                    value = x[layout[attr_name]]
                    lb, ub = attr_info
                    if not lb <= value <= ub:
                        raise ValueError(
                            f"Invalid value for continuous attribute {attr_name} "
                            f"with bounds [{lb}, {ub}]: {value}"
                        )
                case self.AttributeType.CATEGORICAL:
                    value = None
                    for attr_val, i in layout[attr_name].items():
                        if not torch.isclose(
                            x[i], torch.zeros(())
                        ) and not torch.isclose(x[i], torch.ones(())):
                            raise ValueError(
                                f"Invalid one-hot encoding of categorical attribute "
                                f"{attr_name}. Entry is neither 0.0 nor 1.0."
                            )
                        if torch.isclose(x[i], torch.ones(())):
                            if value is not None:
                                raise ValueError(
                                    f"Invalid one-hot encoding of categorical attribute "
                                    f"{attr_name}. Multiple 1.0 entries."
                                )
                            value = attr_val
                case _:
                    raise NotImplementedError()
            decoding.append(value)
        return tuple(decoding)

    def __len__(self):
        """The number of attributes of this input domain."""
        return len(self.__attributes)


class CombinedInputSpace(InputSpace):
    """
    A combined multi-variable input space flattening variables values
    and stacking them in a large vector.
    """

    def __init__(self, variable_domains: OrderedDict[str, InputSpace]):
        """
        Creates a new combined input space from a set of variables.

        :param variable_domains: The variables and their input spaces.
        """
        super().__init__()
        self.__domains = variable_domains
        first_dim_after = []
        i = 0
        for var, domain in variable_domains.items():
            var_size = prod(domain.input_shape)
            i += var_size
            first_dim_after.append(i)
        self.__first_dim_after = tuple(first_dim_after)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.__domains.keys())

    @property
    def offsets(self) -> tuple[int, ...]:
        """
        The first dimensions at which the values of each :code:`variable`
        are located in the combined input space.
        """
        return (0,) + self.__first_dim_after[:-1]

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size((self.__first_dim_after[-1],))

    @property
    def input_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        lbs, ubs = zip(*[d.input_bounds for d in self.__domains.values()])
        lbs = [lb.flatten() for lb in lbs]
        ubs = [ub.flatten() for ub in ubs]
        return torch.hstack(lbs), torch.hstack(ubs)

    def variable_at(self, dim: int) -> str:
        for var, first_dim_after in zip(self.__domains.keys(), self.__first_dim_after):
            if first_dim_after > dim:
                return var

    def domain_of(self, var: str) -> InputSpace:
        return self.__domains[var]

    def dimensions_of(self, var: str) -> int:
        """The number of dimensions occupied by variable :code:`var`"""
        return prod(self.__domains[var].input_shape)

    def local_dim(self, dim: int) -> int:
        """
        Convert a dimension into the dimension of the input space of the variable
        that is located at that dimension.

        :param dim: The dimension to convert.
        :return: The dimension that corresponds to :code:`dim` in
         :code:`in_space.domain_of(in_space.variable_at(dim))`.
        """
        for first_dim_after in self.__first_dim_after:
            if first_dim_after > dim:
                return dim - first_dim_after

    def combine(self, **values: torch.Tensor) -> torch.Tensor:
        """
        Combine the values of the different variables so that
        the result lies in this combined input space.

        Example:
        :code:`input_space.combine(x=torch.tensor([1, 2, 3], y=torch.tensor([4, 5, 6])`
        gives :code:`torch.tensor([1, 2, 3, 4, 5, 6])` if the variable :code:`x`
        comes before :code:`y` in the :class:`CombinedInputSpace` :code:`input_space`.

        :param values: Batches of values of the variables.
        :return: The flattened values, hstacked (:code:`torch.hstack`).
        """
        return torch.hstack([values[var].flatten(1) for var in self.__domains])

    def decompose(self, combined: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Splits a batch of values from the combined input space
        into the values of each variable.
        The values for each variable have the shape of the variable's
        input space.

        :param combined: A batch of values from the combined input space.
        :return: Batches of values of the individual variables.
        """
        values = {}
        prev_first_after = 0
        for i, (var, domain) in enumerate(self.__domains.items()):
            var_first_after = self.__first_dim_after[i]
            var_values = combined[:, prev_first_after:var_first_after]
            values[var] = var_values.reshape((-1,) + domain.input_shape)
            prev_first_after = var_first_after
        return values
