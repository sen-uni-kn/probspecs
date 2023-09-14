# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from enum import Enum, auto, unique

import numpy as np


@unique
class TrinaryLogic(Enum):
    TRUE = auto()
    FALSE = auto()
    UNKNOWN = auto()

    @staticmethod
    def from_bool(value: bool):
        return TrinaryLogic.TRUE if value else TrinaryLogic.FALSE

    @staticmethod
    @np.vectorize
    def not_(value):
        match value:
            case TrinaryLogic.TRUE:
                return TrinaryLogic.FALSE
            case TrinaryLogic.FALSE:
                return TrinaryLogic.TRUE
            case TrinaryLogic.UNKNOWN:
                return TrinaryLogic.UNKNOWN

    @staticmethod
    @np.vectorize
    def and_(*values):
        if any(x is TrinaryLogic.UNKNOWN for x in values):
            return TrinaryLogic.UNKNOWN
        else:
            return TrinaryLogic.from_bool(all(x is TrinaryLogic.TRUE for x in values))

    @staticmethod
    @np.vectorize
    def or_(*values):
        if all(x is TrinaryLogic.UNKNOWN for x in values):
            return TrinaryLogic.UNKNOWN
        else:
            return TrinaryLogic.from_bool(any(x is TrinaryLogic.TRUE for x in values))

    def __eq__(self, other):
        if isinstance(other, TrinaryLogic):
            return self.value == other.value
        elif isinstance(other, bool):
            return self == TrinaryLogic.from_bool(other)
        else:
            return NotImplemented
