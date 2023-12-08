# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from probspecs.operators import Heaviside

import pytest


@pytest.mark.parametrize("auto_lirpa_method", ("ibp", "crown"))
@pytest.mark.parametrize("one_at_zero", (True, False))
def test_heaviside(auto_lirpa_method, one_at_zero):
    module = nn.Sequential(
        Heaviside(one_at_zero),
    )
    bounded_module = BoundedModule(module, torch.empty(1, 5))

    ptb = PerturbationLpNorm(x_L=-torch.ones(10, 5), x_U=torch.ones(10, 5))
    bounded_tensor = BoundedTensor(torch.zeros(10, 5), ptb)

    lbs, ubs = bounded_module.compute_bounds(
        x=(bounded_tensor,), method=auto_lirpa_method
    )
    assert torch.all(lbs <= 0.0)
    assert torch.all(ubs >= 1.0)

    ptb = PerturbationLpNorm(x_L=torch.zeros(10, 5) + 0.01, x_U=torch.ones(10, 5))
    bounded_tensor = BoundedTensor(torch.ones(10, 5), ptb)
    lbs, ubs = bounded_module.compute_bounds(
        x=(bounded_tensor,), method=auto_lirpa_method
    )
    assert torch.all(lbs == 1.0)
    assert torch.all(ubs == 1.0)

    ptb = PerturbationLpNorm(x_L=-torch.ones(10, 5), x_U=torch.zeros(10, 5) - 0.01)
    bounded_tensor = BoundedTensor(torch.ones(10, 5), ptb)
    lbs, ubs = bounded_module.compute_bounds(
        x=(bounded_tensor,), method=auto_lirpa_method
    )
    assert torch.all(lbs == 0.0)
    assert torch.all(ubs == 0.0)
