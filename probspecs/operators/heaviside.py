# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
from torch import nn
from auto_LiRPA import register_custom_op
from auto_LiRPA.operators import Bound


class HeavisideOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, one_at_zero):
        return g.op("custom::Heaviside", x, one_at_zero_i=one_at_zero)

    @staticmethod
    def forward(ctx, x, one_at_zero):
        values = torch.tensor(1.0 if one_at_zero else 0.0)
        return torch.heaviside(x, values)

    @staticmethod
    def backward(ctx, grad_y):
        return torch.zeros_like(grad_y), None


class Heaviside(nn.Module):
    """
    The Heaviside step function (or unit step function).
    The Heaviside function is 0.0 for negative values and 1.0 for positive values.
    For 0.0 it can be other 0.0 or 1.0.
    This is determined by the :code:`one_at_zero` attribute.

    While auto_LiRPA supports the Heaviside function in CROWN, the IBP
    implementation is buggy.
    """

    def __init__(self, one_at_zero=True):
        super().__init__()
        self.one_at_zero = one_at_zero

    def forward(self, x):
        return HeavisideOp.apply(x, self.one_at_zero)


class BoundHeaviside(Bound):
    def __init__(self, attrs, inputs, output_index, options):
        super().__init__(attrs, inputs, output_index, options)
        self.one_at_zero = attrs["one_at_zero"]
        self.use_default_ibp = True

    def forward(self, x):
        return HeavisideOp.apply(x, self.one_at_zero)

    def bound_backward(
        self, last_lA, last_uA, *x, start_node=None, start_shape=None, **kwargs
    ):
        # Based on https://github.com/Verified-Intelligence/auto_LiRPA/blob/d1296708f74064472448a388f15cef43cf6cef31/auto_LiRPA/operators/activations.py
        x = x[0]
        if x is not None:
            x_lb = x.lower
            x_ub = x.upper
        else:
            x_lb = self.lower
            x_ub = self.upper

        if self.one_at_zero:
            all_zero = x_ub < 0.0
            all_one = x_lb >= 0.0
        else:
            all_zero = x_ub <= 0.0
            all_one = x_lb > 0.0
        lower_b = all_one.to(dtype=x_lb.dtype, device=x_lb.device)
        upper_b = 1.0 - all_zero.to(dtype=x_lb.dtype, device=x_lb.device)

        # We use an adaptive slope as in the ReLU bound of CROWN/DeepPoly.
        # Slope lower bound (when x_lb < 0 < x_ub):
        #  - if |x_lb| >= |x_ub|, slope = 0.
        #  - otherwise, slope = 1/ub.
        # If x >= 0 or x <= 0, slope is always 0.
        neg_part_is_larger = -x_lb >= x_ub
        lower_d = torch.where(
            all_zero | all_one | neg_part_is_larger,
            0.0,
            1.0 / x_ub.clamp(min=1e-3),  # avoid overly steep slopes
        ).to(dtype=x_lb.dtype, device=x_lb.device)
        # Slope upper bound (when x_lb < 0 < x_ub):
        #  - if |x_lb| < |x_ub|, slope = 0.
        #  - otherwise, slope = -1/lb
        upper_d = torch.where(
            all_zero | all_one | ~neg_part_is_larger,
            0.0,
            1.0 / (-x_lb).clamp(min=1e-3),
        ).to(dtype=x_lb.dtype, device=x_lb.device)
        lower_d = lower_d.unsqueeze(0)
        upper_d = upper_d.unsqueeze(0)

        # Choose upper or lower bounds based on the sign of last_A
        uA = lA = None
        ubias = lbias = 0.0
        if last_uA is not None:
            neg_uA = last_uA.clamp(max=0)
            pos_uA = last_uA.clamp(min=0)
            uA = upper_d * pos_uA + lower_d * neg_uA
            ubias = (pos_uA * upper_b + neg_uA * lower_b).flatten(2).sum(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lower_d * pos_lA
            lbias = (neg_lA * upper_b + pos_lA * lower_b).flatten(2).sum(-1)
        return [(lA, uA), (None, None)], lbias, ubias


register_custom_op("custom::Heaviside", BoundHeaviside)
