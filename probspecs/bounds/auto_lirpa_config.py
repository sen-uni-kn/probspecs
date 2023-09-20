# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from dataclasses import dataclass


@dataclass
class AutoLiRPAConfig:
    method: str
    bound_ops: dict
