# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from dataclasses import dataclass

from frozendict import frozendict


@dataclass
class AutoLiRPAParams:
    """
    Parameters for auto_LiRPA.

     - auto_lirpa_method: The :code:`auto_LiRPA` bound propagation method
       to use for computing bounds.
       More details in the :func:`auto_LiRPA.BoundedModule.compute_bounds` documentation.
     - bound_ops: :code:`auto_LiRPA` bound propagation options.
       More details in the :func:`auto_LiRPA.BoundedModule` documentation.
    """

    method: str = "alpha-CROWN"
    bound_ops: dict | frozendict = frozendict(
        {"optimize_bound_args": frozendict({"iteration": 20, "lr_alpha": 0.1})}
    )
