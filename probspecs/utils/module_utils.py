# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
"""
Utilities for :code:`torch.nn.Module`.
"""
from typing import Callable

import torch


class WrapperModule(torch.nn.Module):
    """
    Wraps an arbitrary callable as a :code:`torch.nn.Module.
    Centrally, it allows to register :code:`torch.nn.Modules` that are used
    in the callable as submodules.

    Registering submodules is central for computing gradients, including turning
    off gradient computation.
    """

    def __init__(
        self,
        target: Callable[..., torch.Tensor],
        submodules: dict[str, torch.nn.Module] | list[torch.nn.Module],
    ):
        """
        Create a wrapper module for the callable :code:`target`.

        :param target: The callable to wrap.
        :param submodules: :code:`torch.nn.Modules` that are used in :code:`target`.
         These are registered as submodules to this :class:`WrapperModule`.
        """
        super().__init__()
        if isinstance(submodules, list):
            self.submodules = torch.nn.ModuleList(submodules)
        elif isinstance(submodules, dict):
            self.submodules = torch.nn.ModuleDict(submodules)
        else:
            raise ValueError(
                f"submodules need to be list or dict. Got {type(submodules)}"
            )
        self.target = target

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)
