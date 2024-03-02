# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Union

import numpy as np
import torch

__all__ = ["TrinaryLogic"]


class TrinaryLogic(int):
    """
    :code:`TrinaryLogic` is an extension of booleans (`True` and `False`)
    that includes a third value: `Unknown`.

    :code:`TrinaryLogic` extends int, so that it's values can be stored in tensors.
    Torch will convert the :code:`TrinaryLogic` values into ints.
    Be sure to use the class methods of :code:`TrinaryLogic` when working with
    such tensors.
    While operators such as :code:`-`, :code:`+`, :code:`*`, :code:`>`, and more
    are available, it is recommended not to use them to manipulate TrinaryLogic objects.
    Use class methods like :code:`TrinaryLogic.not_`, :code:`TrinaryLogic.or_`,
    :code:`TrinaryLogic.and_`, ... instead.
    Similarly, it is not recommended to create new :code:`TrinaryLogic` instances.
    Use the constants :code:`TrinaryLogic.TRUE`, :code:`TrinaryLogic.FALSE`
    and :code:`TrinaryLogic.UNKNOWN` instead.
    """

    TRUE = ...
    FALSE = ...
    UNKNOWN = ...

    def __new__(cls, value, *args, **kwargs):
        value = np.sign(value)
        return super(cls, cls).__new__(cls, value)

    @staticmethod
    def from_bool(value: bool):
        return TrinaryLogic.TRUE if value else TrinaryLogic.FALSE

    @staticmethod
    def is_true(
        value: Union["TrinaryLogic", np.ndarray, torch.Tensor],
    ) -> bool | np.ndarray | torch.Tensor:
        return value > 0

    @staticmethod
    def is_false(
        value: Union["TrinaryLogic", np.ndarray, torch.Tensor],
    ) -> bool | np.ndarray | torch.Tensor:
        return value < 0

    @staticmethod
    def is_unknown(
        value: Union["TrinaryLogic", np.ndarray, torch.Tensor],
    ) -> bool | np.ndarray | torch.Tensor:
        return value == 0

    @staticmethod
    def not_(value: Union["TrinaryLogic", np.ndarray, torch.Tensor]):
        """
        Negates a :code:`TrinaryLogic` instance or the elements of an array or tensor
        of :code:`TrinaryLogic` instances (or corresponding integers).
        """
        return -value

    @staticmethod
    def and_(*values: Union["TrinaryLogic", np.ndarray, torch.Tensor]):
        """
        And-connects several :code:`TrinaryLogic` instances or the elements of several
        arrays or tensors of :code:`TrinaryLogic` instances (or corresponding integers).
        """
        if len(values) == 0:
            return TrinaryLogic.TRUE

        any_unknown = values[0] == 0
        all_true = values[0] == 1
        for value in values[1:]:
            any_unknown |= value == 0
            all_true &= value == 1
        # 2 * all_true - 1 maps True/False to 1/-1
        # Multiplying by ~any_unknown sets all entries to zero (Unknown)
        # where one of values contains Unknown
        return ~any_unknown * (2 * all_true - 1)

    @staticmethod
    def or_(*values: Union["TrinaryLogic", np.ndarray, torch.Tensor]):
        """
        Or-connects several :code:`TrinaryLogic` instances or the elements of several
        arrays or tensors of :code:`TrinaryLogic` instances (or corresponding integers).
        """
        if len(values) == 0:
            return TrinaryLogic.FALSE

        any_unknown = values[0] == 0
        any_true = values[0] == 1
        for value in values[1:]:
            any_unknown |= value == 0
            any_true |= value == 1
        # see TrinaryLogic.and_
        return ~any_unknown * (2 * any_true - 1)

    def __eq__(self, other):
        if isinstance(other, bool):
            return self == TrinaryLogic.from_bool(other)
        elif isinstance(other, int | np.ndarray):
            return np.sign(other) == self
        elif isinstance(other, torch.Tensor):
            return torch.sign(other) == self
        elif isinstance(other, TrinaryLogic):
            return super().__eq__(other)
        else:
            return NotImplemented

    def __str__(self):
        if self > 0:
            return "True"
        elif self < 0:
            return "False"
        else:
            return "Unknown"

    def __repr__(self):
        return str(self)


TrinaryLogic.TRUE = TrinaryLogic(1)
TrinaryLogic.FALSE = TrinaryLogic(-1)
TrinaryLogic.UNKNOWN = TrinaryLogic(0)
