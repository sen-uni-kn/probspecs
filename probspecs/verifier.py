# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import os
from enum import Enum, auto, unique
from math import ceil
from typing import Any, Callable
from time import time
import multiprocessing as mp

import torch
from frozendict import frozendict

from .bounds.auto_lirpa_params import AutoLiRPAParams
from .probability_distribution import ProbabilityDistribution
from .input_space import InputSpace
from .formula import (
    Composition,
    Formula,
    Probability,
    Function,
    Inequality,
    Expression,
    ExternalFunction,
    ExternalVariable,
    ExplicitFunction,
)
from .trinary_logic import TrinaryLogic as TL
from .bounds.probability_bounds import probability_bounds
from .bounds.network_bounds import network_bounds
from .utils.formula_utils import collect_requires_bounds


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
    formula: Formula | Inequality,
    externals: dict[str, Callable | torch.Tensor],
    external_vars_domains: dict[str, InputSpace],
    external_vars_distributions: dict[str, ProbabilityDistribution],
    workers: tuple[str | torch.device, ...] | None = None,
    timeout: float | None = None,
) -> tuple[VerificationStatus, dict[Function, tuple[torch.Tensor, torch.Tensor]]]:
    """
    Verifies a formula.

    :param formula: The formula to verify.
    :param externals: Keyword arguments for evaluating the formula,
     for example, neural networks that appear in the formula.
     See :class:`ExternalFunction` and :class:`ExternalVariable`.
    :param external_vars_domains: :class:`InputSpace` objects for all
     external variables (:class:`ExternalVariable` objects and arguments of
     :class:`ExternalFunction` objects)
     appearing in :code:`formula`.
     These are used for computing bounds on the external functions
     (for example, neural networks) and probabilities in the formula.
    :param external_vars_distributions: The distributions of
     the external variables (:class:`ExternalVariable` objects and arguments of
     :class:`ExternalFunction` objects)
     appearing in :code:`formula`.
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
    if workers is None:
        # TODO: use GPUs by default
        workers = ("cpu",) * len(os.sched_getaffinity(0))

    # Replace compositions of functions by a single function computing
    # the composition
    formula, compose_substitutions = fuse_compositions(formula)

    requires_bounds = collect_requires_bounds(formula)
    bounded_terms_substitution = {
        term: ExternalVariable(f"?{term}?") for term in requires_bounds
    }
    term_subs = {term: var.name for term, var in bounded_terms_substitution.items()}
    top_level_formula = formula.replace(bounded_terms_substitution)

    bounds_queue = mp.SimpleQueue()
    worker_processes = []
    try:
        terms_per_worker = ceil(len(requires_bounds) / len(workers))
        remaining_terms = list(requires_bounds)
        for i, worker_device in enumerate(workers):
            if len(remaining_terms) == 0:
                break
            worker_terms = remaining_terms[:terms_per_worker]
            remaining_terms = remaining_terms[terms_per_worker:]
            worker = mp.Process(
                target=_compute_bounds_worker,
                args=worker_terms,
                kwargs={
                    "externals": externals,
                    "external_vars_domains": external_vars_domains,
                    "external_vars_distributions": external_vars_distributions,
                    "results_queue": bounds_queue,
                    "device": worker_device,
                },
            )
            worker_processes.append(worker)
            worker.start()

        def terminate_workers():
            for worker_ in worker_processes:
                worker_.terminate()

        start_time = time()
        best_bounds = {}
        # fetch bounds until we have a bound for every term in requires_bounds
        while len(best_bounds) < len(requires_bounds):
            term, new_bounds = bounds_queue.get()
            best_bounds[term_subs[term]] = new_bounds

        outcome = top_level_formula.propagate_bounds(**best_bounds)
        while outcome == TL.UNKNOWN:
            if timeout is not None and time() > start_time + timeout:
                terminate_workers()
                raise VerificationTimeout()

            term, new_bounds = bounds_queue.get()
            best_bounds[term_subs[term]] = new_bounds
            outcome = top_level_formula.propagate_bounds(**best_bounds)

        terminate_workers()
        best_bounds = {term: best_bounds[term_subs[term]] for term in requires_bounds}
        return VerificationStatus.from_(outcome), best_bounds
    finally:
        if len(worker_processes) > 0:
            for worker in worker_processes:
                worker.terminate()
        bounds_queue.close()


def _compute_bounds_worker(
    *target_terms: Function,
    externals: dict[str, Callable | torch.Tensor],
    external_vars_domains: dict[str, InputSpace],
    external_vars_distributions: dict[str, ProbabilityDistribution],
    results_queue: mp.Queue,
    device: torch.device | str,
):
    """
    A worker for computing bounds on one or multiple target terms.
    Workers are intended to be run in parallel.
    However, one worker can compute bounds for several terms.
    These bounds are updated sequentially by the worker.

    :param target_terms: The terms (probabilities, networks)
     for which to compute bounds.
    :param externals: Values for external variables and function for evaluating
     the target terms.
    :param external_vars_bounds: Bounds for external variables.
    :param external_vars_distributions: Probability distributions for external
     variables.
    :param results_queue: The :code:`multiprocessing.Queue` for posting results.
    :param device: Which device to use for computing bounds.
    """
    external_callables = {
        name: e for name, e in externals.items() if isinstance(e, Callable)
    }

    def get_bounds_gen(term: Function):
        if isinstance(term, Probability):
            return probability_bounds(
                probability=term,
                networks=external_callables,
                variable_domains=external_vars_domains,
                variable_distributions=external_vars_distributions,
                auto_lirpa_params=AutoLiRPAParams(method="CROWN"),  # FIXME
                split_heuristic="longest-edge",  # FIXME
            )  # TODO: further config
        elif isinstance(term, ExternalFunction):
            network = term.get_function(**externals)
            if len(term.arg_names) != 1:
                raise ValueError(
                    f"Currently, only external functions with a single argument "
                    f"are supported. Function has multiple arguments: {term}"
                )
            else:
                arg_name = term.arg_names[0]
                arg_bounds = external_vars_domains[arg_name].input_bounds
            return network_bounds(network, input_bounds=arg_bounds)

    bounds_gens = [get_bounds_gen(term) for term in target_terms]
    while True:
        for term, bounds_gen in zip(target_terms, bounds_gens):
            new_bounds = next(bounds_gen)
            results_queue.put((term, new_bounds))


def fuse_compositions(
    term: Formula | Inequality | Expression | Function,
) -> tuple[
    Formula | Inequality | Expression | Function, dict[Composition, ExplicitFunction]
]:
    """
    Replaces all function compositions (:class:`Compose` instances)
    with new external functions that evaluate the composition.

    :param term: The term in which to fuse the compositions.
    :return:
     - A new term with the same top-level structure, but compositions
       replaced by new external functions.
     - A mapping from compositions to the new external functions they
       are replaced with.
    """
    compositions = term.collect(lambda sub: isinstance(sub, Composition))

    # Fuse sub-compositions first to simplify collecting arguments.
    # Reason: for compositions inside compositions the arguments of the
    # Compose.func attributes may not be collected as arguments of the
    # function with which the top composition is replaced.
    compositions = {
        compose: Composition(
            compose.func,
            frozendict(
                {arg: fuse_compositions(expr)[0] for arg, expr in compose.args.items()}
            ),
        )
        for compose in compositions
    }

    def replacement(compose: Composition) -> ExternalFunction:
        def is_external(term_):
            return isinstance(term_, ExternalVariable | ExternalFunction)

        def get_args(external: ExternalVariable | ExternalFunction):
            if isinstance(external, ExternalVariable):
                return (external.name,)
            elif isinstance(external, ExternalFunction):
                if isinstance(external, ExplicitFunction):
                    return external.arg_names
                else:
                    return (external.func_name,) + external.arg_names

        externals = compose.collect(is_external)
        compose_args = sum((get_args(e) for e in externals), ())
        if isinstance(compose.func, ExternalFunction) and not isinstance(
            compose.func, ExplicitFunction
        ):
            compose_args += (compose.func.func_name,)
        compose_args = tuple(set(compose_args))

        def eval_compose(*args):
            kwargs = {
                name: value for name, value in zip(compose_args, args, strict=True)
            }
            return compose(**kwargs)

        return ExplicitFunction(f"?{compose}?", compose_args, eval_compose)

    substitution = {
        orig_compose: replacement(fused_compose)
        for orig_compose, fused_compose in compositions.items()
    }
    return term.replace(substitution), substitution
