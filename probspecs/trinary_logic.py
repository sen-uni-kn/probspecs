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

        any_false = values[0] == TrinaryLogic.FALSE
        all_true = values[0] == TrinaryLogic.TRUE
        for value in values[1:]:
            any_false |= value == TrinaryLogic.FALSE
            all_true &= value == TrinaryLogic.TRUE
        # True==1.0, False==0.0
        # Three cases (all_true=True and any_false=True can not appear):
        #  - all_true=True, any_false=False: 1 - 0 = 1 (TL.TRUE)
        #  - all_true=False, any_false=True: 0 - 1 = -1 (TL.FALSE)
        #  - all_true=False, any_false=False: 0 - 0 = 0 (TL.UNKNOWN)
        # + 0 just converts bool to int
        return (all_true + 0) - (any_false + 0)

    @staticmethod
    def or_(*values: Union["TrinaryLogic", np.ndarray, torch.Tensor]):
        """
        Or-connects several :code:`TrinaryLogic` instances or the elements of several
        arrays or tensors of :code:`TrinaryLogic` instances (or corresponding integers).
        """
        if len(values) == 0:
            return TrinaryLogic.FALSE

        all_false = values[0] == TrinaryLogic.FALSE
        any_true = values[0] == TrinaryLogic.TRUE
        for value in values[1:]:
            all_false &= value == TrinaryLogic.FALSE
            any_true |= value == TrinaryLogic.TRUE
        # see TrinaryLogic.and_
        # Three cases:
        #  - any_true=True, all_false=False: 1 - 0 = 1 (TL.TRUE)
        #  - any_true=False, all_false=True: 0 - 1 = -1 (TL.FALSE)
        #  - any_true=False, all_false=False: 0 - 0 = 0 (TL.UNKNOWN)
        return (any_true + 0) - (all_false + 0)

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
