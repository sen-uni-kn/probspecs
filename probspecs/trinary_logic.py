# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from enum import Enum, auto, unique

import numpy as np


@unique
class TrinaryLogic(Enum):
    TRUE = auto()
    FALSE = auto()
    UNKNOWN = auto()

    @np.vectorize
    def not_(self):
        match self:
            case TrinaryLogic.TRUE:
                return TrinaryLogic.FALSE
            case TrinaryLogic.FALSE:
                return TrinaryLogic.TRUE
            case TrinaryLogic.UNKNOWN:
                return TrinaryLogic.UNKNOWN

    @np.vectorize
    def and_(self, *others):
        values = (self, *others)
        if any(x is TrinaryLogic.UNKNOWN for x in values):
            return TrinaryLogic.UNKNOWN
        else:
            return all(x is TrinaryLogic.TRUE for x in values)

    @np.vectorize
    def or_(self, *others):
        values = (self, *others)
        if all(x is TrinaryLogic.UNKNOWN for x in values):
            return TrinaryLogic.UNKNOWN
        else:
            return any(x is TrinaryLogic.TRUE for x in values)
