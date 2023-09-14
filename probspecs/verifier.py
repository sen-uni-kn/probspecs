# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from enum import Enum, auto, unique
from math import ceil
from typing import Any, Callable
from time import time
import multiprocessing as mp

import torch

from .formula import (
    Formula,
    Probability,
    Function,
    Inequality,
    Expression,
    ElementAccess,
    ExternalFunction,
    ExternalVariable,
)
from .trinary_logic import TrinaryLogic as TL


@unique
class VerificationStatus(Enum):
    SATISFIED = auto()
    VIOLATED = auto()
    UNKNOWN = auto()
    ERROR = auto()

    @staticmethod
    def from_(val: TL):
        match val:
            case TL.TRUE:
                return VerificationStatus.SATISFIED
            case TL.FALSE:
                return VerificationStatus.VIOLATED
            case TL.UNKNOWN:
                return VerificationStatus.UNKNOWN


class VerificationTimeout(Exception):
    pass


def verify(
    formula: Formula,
    externals: dict[str, Callable | torch.Tensor],
    external_vars_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    external_vars_distributions: dict[str, Any],  # TODO: type
    workers: tuple[str | torch.device, ...],
    timeout: float | None = None,
) -> tuple[VerificationStatus, dict[Function, tuple[torch.Tensor, torch.Tensor]]]:
    """
    Verifies a formula.

    :param formula: The formula to verify.
    :param externals: Keyword arguments for evaluating the formula,
     for example, neural networks that appear in the formula.
     See :class:`ExternalFunction` and :class:`ExternalVariable`.
    :param external_vars_bounds: Bounds on the :class:`ExternalVariable`
     objects in :code:`formula`.
     These are used for computing bounds on the external functions
     (for example, neural networks) and probabilities in the formula.
    :param external_vars_distributions: The distributions of
     the external variables.
     These are used for computing bounds on probabilities.
    :param workers: The devices (cpu, cuda:1, cuda:2, ...)
     to use for the different workers computing bounds.
     The length of this tuple determines the number of workers.
    :param timeout: The timeout for verification.
     When verification exceeds this time budget, it is aborted.
    :return: The status of verification and a verification witness.
     The verification witness is a set of bounds for functions in :code:`formula`
     that allows to prove/disprove :code:`formula`.
    """

    def collect_requires_bounds(
        obj: Formula | Inequality | Expression | Function,
    ) -> tuple[Function, ...]:
        match obj:
            case Formula(_, children) | Expression(_, children):
                return sum(
                    (collect_requires_bounds(child) for child in children),
                    start=(),
                )
            case Inequality():
                lhs_result = collect_requires_bounds(obj.lhs)
                rhs_result = collect_requires_bounds(obj.rhs)
                return lhs_result + rhs_result
            case ElementAccess():
                return collect_requires_bounds(obj.source)
            case Probability() | ExternalFunction() | ExternalVariable():
                return (obj,)

    requires_bounds = collect_requires_bounds(formula)
    bounded_terms_substitution = {
        term: ExternalVariable(str(term)) for term in requires_bounds
    }
    top_level_formula = formula.replace(bounded_terms_substitution)

    with mp.SimpleQueue() as bounds_queue:
        terms_per_worker = ceil(len(requires_bounds) / len(workers))
        workers = []
        for i, worker_device in enumerate(workers):
            worker = mp.Process(
                target=_compute_bounds_worker,
                args=requires_bounds[i*terms_per_worker:(i+1)*terms_per_worker],
                kwargs={
                    "externals": externals,
                    "external_vars_bounds": external_vars_bounds,
                    "external_vars_distributions": external_vars_distributions,
                    "results_queue": bounds_queue,
                    "device": worker_device,
                },
            )
            workers.append(worker)
            worker.start()

        best_bounds = {}
        # fetch bounds until we have a bound for every term in requires_bounds
        while set(best_bounds.keys()).issuperset(requires_bounds):
            term, new_bounds = bounds_queue.get()
            best_bounds[str(term)] = new_bounds

        start_time = time()
        outcome = top_level_formula.propagate_bounds(**best_bounds)
        while outcome is TL.UNKNOWN:
            if time() > start_time + timeout:
                for worker in workers:
                    worker.terminate()
                raise VerificationTimeout()

            term, new_bounds = bounds_queue.get()
            best_bounds[str(term)] = new_bounds
            outcome = top_level_formula.propagate_bounds(**best_bounds)

        best_bounds = {term: best_bounds[str(term)] for term in requires_bounds}
        return VerificationStatus.from_(outcome), best_bounds


def _compute_bounds_worker(
    *target_terms: Function,
    externals: dict[str, Callable | torch.Tensor],
    external_vars_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    external_vars_distributions: dict[str, Any],  # TODO: type
    results_queue: mp.Queue,
    device: torch.device | str
):
    """
    A worker for computing bounds on one or multiple target terms.
    Workers are intended to be run in parallel.
    However, one worker can compute bounds for several terms.
    These bounds are updated sequentially by the worker.

    :param target_terms: The terms (probabilities, networks)
     for which to compute bounds.
    :param device: Which device to use for computing bounds.
    """
    pass
