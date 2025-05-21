#  Copyright (c) 2024 Annonymous Authors
#  Licensed under the MIT License
import argparse
from time import time
from pathlib import Path

import onnx
from onnx2pytorch import ConvertModel
import torch

from probspecs import (
    prob,
    Verifier,
    ExternalFunction,
    ExternalVariable,
    TensorInputSpace,
    Formula,
)
from probspecs.distributions import Uniform, ProbabilityDistribution
from probspecs.utils.tensor_utils import TENSOR_LIKE
from probspecs.utils.yaml import yaml
from experiments.utils import log_machine_and_code_details


class VCASUniform(ProbabilityDistribution):
    """Sets the 3rd dimension to a constant."""

    def __init__(
        self,
        support: tuple[TENSOR_LIKE, TENSOR_LIKE],
        dtype: torch.dtype = torch.double,
    ):
        lbs, ubs = support
        assert torch.isclose(lbs[2], ubs[2])
        self.__intruder_climbing_rate = lbs[2]
        self.__base = Uniform((lbs[[0, 1, 3]], ubs[[0, 1, 3]]), dtype)

    def _remove_intruder_climbing_rate(
        self, *tensors: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        def select(t: torch.Tensor) -> torch.Tensor:
            if t.shape[-1] == 3:
                return t.reshape((-1, 3))
            else:
                return t.reshape((-1,) + self.event_shape)[:, [0, 1, 3]]

        ts = [select(t) for t in tensors]
        if len(tensors) == 1:
            return ts[0]
        else:
            return tuple(ts)

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.__base.probability(self._remove_intruder_climbing_rate(*event))

    def density(self, elementary: torch.Tensor) -> torch.Tensor:
        return self.__base.density(self._remove_intruder_climbing_rate(elementary))

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        sample = torch.empty((num_samples,) + self.event_shape, dtype=self.dtype)
        sample[:, [0, 1, 3]] = self.__base.sample(num_samples, seed)
        sample[:, 2] = self.__intruder_climbing_rate
        return sample

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([4])

    @property
    def dtype(self) -> torch.dtype:
        return self.__base.dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self.__base.dtype = dtype

    @property
    def parameters(self) -> torch.Tensor:
        return self.__base.parameters

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        self.__base.parameters = parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Quantify Violations of VCAS Safety Specifications"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="A timeout for computing bounds on the frequency of property violations "
        "in seconds.",
    )
    parser.add_argument(
        "--probability-bounds-config",
        default="{}",
        help="A configuration for computing bounds. Can be a path to a YAML file "
        "or a yaml string. Have a look at the ProbabilityBounds class for details "
        "on which configurations are available.",
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether to print progress messages."
    )
    args = parser.parse_args()

    print("Running Experiment: VCAS - Safety")
    print("=" * 100)
    print("Command Line Arguments:")
    print(args)
    log_machine_and_code_details()

    onnx_network = onnx.load("resources/vcas/VCAS1.onnx")
    network = ConvertModel(onnx_network)

    input_lbs = torch.tensor([-8000.0, 0.0, -30.0, 0.0])
    input_ubs = torch.tensor([0.0, 100.0, -30.0, 40.0])
    input_space = TensorInputSpace(input_lbs, input_ubs)

    x = ExternalVariable("x")
    # x[:, 0]: h
    # x[:, 1]: \dot{h_A}
    # x[:, 2]: \dot{h_B} (fixed)
    # x[:, 3]: t
    net_func = ExternalFunction("network", ("x",))
    # net_func[:, 0]: score for Clear-of-Conflict (CoC)
    # net_func[:, 1:8]: scores for other classes
    # Unlike ACAS XU, the classifier chooses the class with the *maximal* score.
    # The score for CoC has to be maximal.
    output_constraint = Formula(
        Formula.Operator.AND,
        tuple(net_func[:, 0] >= net_func[:, i] for i in range(1, 8)),
    )
    target_formula = prob(output_constraint) >= 0.9
    # count violations: uniform distribution
    input_distribution = VCASUniform(input_space.input_bounds)

    timeout = args.timeout
    if timeout is None:
        timeout = float("inf")

    if "{" in args.probability_bounds_config or "\n" in args.probability_bounds_config:
        prob_bounds_config = args.probability_bounds_config
    else:
        prob_bounds_config = Path(args.probability_bounds_config)
    prob_bounds_config = yaml.load(prob_bounds_config)
    prob_bounds_config = {
        "batch_size": 1024,
        "log": args.log,
    } | prob_bounds_config
    print("prob_bounds_config", prob_bounds_config)
    verifier = Verifier(
        worker_devices="cpu",
        timeout=timeout,
        log=args.log,
        probability_bounds_config=prob_bounds_config,
        parallel=False,
    )

    print("Starting Verification.")
    start_time = time()
    verification_status, probability_bounds = verifier.verify(
        target_formula,
        {"network": network},
        {"x": input_space},
        {"x": input_distribution},
    )
    end_time = time()
    print(verification_status)
    print(probability_bounds)
    print(f"Runtime: {end_time-start_time:.4f}s")
