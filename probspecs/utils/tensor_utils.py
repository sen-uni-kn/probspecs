#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
from typing import Sequence

import numpy as np
import torch

__all__ = ["TENSOR_LIKE", "to_tensor", "copy_to_tensor"]


# exclude tuples because of potential risk of confusion with bound tuples
TENSOR_LIKE = (
    torch.Tensor
    | np.ndarray
    | list[float | int]
    | list[list[float | int]]
    | list[list[list]]
    | float
    | int
)


def to_tensor(
    array_like: TENSOR_LIKE, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Transforms an array like (torch.Tensor, np.ndarray, list, tuple, ...)
    into a tensor.
    Tensors are returned as-is.
    Numpy arrays are converted using `torch.as_tensor`.
    The tensor uses the same storage as the numpy array.
    Other objects are converted using `torch.tensor`.
    """
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype)
    elif isinstance(array_like, np.ndarray):
        return torch.as_tensor(array_like, dtype=dtype)
    else:
        return torch.tensor(array_like, dtype=dtype)


def copy_to_tensor(array_like: TENSOR_LIKE) -> torch.Tensor:
    """
    Transforms an array like (torch.Tensor, np.ndarray, list, tuple, ...)
    into a tensor.
    Tensors as input are copied.
    Equally so for numpy arrays.
    The returned tensor won't refer to the same storage.
    Data in lists and tuples is copied anyway when creating a tensor.
    """
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().clone()
    elif isinstance(array_like, np.ndarray):
        return torch.as_tensor(array_like).clone()
    else:
        return torch.tensor(array_like)
