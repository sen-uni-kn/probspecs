# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class FactorAnalysisModel(nn.Linear):
    """
    A population model for factor analysis.

    Takes a concatenation of factor values and noise as input
    and returns a linear combination of the factors summed with the
    noise for each feature.
    """

    def __init__(self, factor_loadings: torch.Tensor):
        """
        Create a :code:`FactorAnalysisModel`.

        :param factor_loadings: The factor loading matrix
         of shape :code:`(n_features, n_factors)`.
        """
        n_features, n_factors = factor_loadings.shape
        super().__init__(n_factors + n_features, n_features, bias=False)
        weight = torch.zeros((n_features, n_factors + n_features))
        weight[:, :n_factors] = factor_loadings
        for i in range(n_features):
            weight[i, n_factors + i] = 1.0
        self.weight = nn.Parameter(weight)


class EncodeOneHot(nn.Module):
    """
    Transforms certain features given as numerical values into a one-hot
    encoding using linear transformations and :code:`torch.heaviside`.
    Other features are left as-is.
    """

    def __init__(self, num_values: Sequence[int | None]):
        """
        Create a new :code:`EncodeOneHot` instance.

        :param num_values: The number of values of each categorical
         attribute in the order they appear in the purely numerical input space.
         Other attributes are marked with :code:`None`.
        """
        super().__init__()
        input_size = len(num_values)
        output_size = sum(1 if num is None else num for num in num_values)
        # copy the other attributes over to the output space
        # while zeroing out all elements where one-hot encodings are placed
        zero_out_one_hot = torch.zeros(output_size, input_size)
        k = 0
        for i, num in enumerate(num_values):
            if num is None:
                zero_out_one_hot[k, i] = 1.0
                k += 1
            else:
                k += num
        self.zero_out_one_hot = nn.Parameter(zero_out_one_hot)

        # Copy the numerically-encoded categorical attributes to the output space.
        # For a numerical value 0 < a < max of a categorical attribute, bias_lower is
        # a - 0.5 and bias_upper is a + 0.5, while weight_lower and weight_upper are
        # both 1.0.
        # Then, h(weight_lower * x - bias_lower) - h(weight_upper * x - bias_upper) is 1
        # iff x = a (h is heaviside).
        # The values a = 0 and a = max (the highest value) need special treatment.
        # We want to produce 1 for a = 0 for all x <= 0.5.
        # For this, we set weight_lower for a = 0 to -1 and bias_lower to -0.5.
        # To make h(weight_upper x - bias_upper) always zero for a = 0, we set
        # weight_upper to 0.0 and bias_upper to 1.
        # For a = max, we set weight_lower to 1.0 and bias_lower to max - 0.5.
        # We make the second heaviside term zero always, as we do for a = 0.
        one_hot_weight_lower = torch.zeros(output_size, input_size)
        one_hot_weight_upper = torch.zeros(output_size, input_size)
        one_hot_bias_lower = torch.zeros(output_size)
        one_hot_bias_upper = torch.zeros(output_size)
        k = 0
        for i, num in enumerate(num_values):
            if num is not None:
                one_hot_weight_lower[k, i] = -1.0  # for a = 0
                one_hot_weight_upper[k, i] = 0.0
                one_hot_bias_lower[k] = -0.5
                one_hot_bias_upper[k] = 1.0
                k += 1
                for j in range(1, num - 1):
                    one_hot_weight_lower[k, i] = 1.0
                    one_hot_weight_upper[k, i] = 1.0
                    one_hot_bias_lower[k] = j - 0.5
                    one_hot_bias_upper[k] = j + 0.5
                    k += 1
                one_hot_weight_lower[k, i] = 1.0  # for a = max
                one_hot_weight_upper[k, i] = 0.0
                one_hot_bias_lower[k] = num - 1 - 0.5
                one_hot_bias_upper[k] = 1.0
                k += 1
            else:
                k += 1
        self.one_hot_weight_lower = nn.Parameter(one_hot_weight_lower)
        self.one_hot_weight_upper = nn.Parameter(one_hot_weight_upper)
        self.one_hot_bias_lower = nn.Parameter(one_hot_bias_lower)
        self.one_hot_bias_upper = nn.Parameter(one_hot_bias_upper)
        self.one = torch.ones(())

    def __call__(self, x):
        others = F.linear(x, self.zero_out_one_hot)
        one_hot_lower = F.linear(x, self.one_hot_weight_lower, -self.one_hot_bias_lower)
        one_hot_upper = F.linear(x, self.one_hot_weight_upper, -self.one_hot_bias_upper)
        one_hot_lower = torch.heaviside(one_hot_lower, self.one)
        one_hot_upper = torch.heaviside(one_hot_upper, self.one)
        one_hot = one_hot_lower - one_hot_upper
        return others + one_hot


class Normalize(nn.Module):
    """
    Applies z-score normalization.

    Subtracts a fixed mean vector and divides by a fixed standard deviation vector.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class Denormalize(nn.Module):
    """
    Reverts a z-score normalization.

    Multiplies by a fixed standard deviation vector and adds a fixed mean vector.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class Identity(nn.Linear):
    """
    An identity transformation.
    Serves for avoiding issues with AutoLiRPA by providing
    explicit shape information.
    """

    def __init__(self, num_features: int):
        super().__init__(num_features, num_features, bias=False)
        self.weight = nn.Parameter(torch.eye(num_features))
