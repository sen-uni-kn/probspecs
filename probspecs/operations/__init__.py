# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
"""
This package contains alternative implementations of common operations that are
unsupported by :code:`auto_LiRPA` as they are, but can be computed using only
supported operations.

An example is :code:`torch.clamp`, which is not supported by :code:`auto_LiRPA`,
but can be computed using only `ReLU` and addition and subtraction.
"""
from .clamp import clamp, minimum, maximum
