# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from enum import Enum, auto, unique
from math import ceil
from typing import Any, Callable
from time import time
import multiprocessing as mp

import torch
from frozendict import frozendict

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
    # Replace compositions of functions by a single function computing
    # the composition
    formula = fuse_compositions(formula)

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
                args=requires_bounds[i * terms_per_worker : (i + 1) * terms_per_worker],
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

    def get_bounds_gen(term: Function):
        if isinstance(term, Probability):
            return probability_bounds(
                probability=term,
                networks=None,  # TODO
            )
        elif isinstance(term, ExplicitFunction):
            return network_bounds(
                term.func,
                input_bounds=None,  # TODO
            )
        elif isinstance(term, ExternalFunction):
            return network_bounds(
                externals[term.func_name],
                input_bounds=None,  # TODO
            )

    bounds_gens = [get_bounds_gen(term) for term in target_terms]
    while True:
        for term, bounds_gen in zip(target_terms, bounds_gens, strict=True):
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

        return ExplicitFunction(f"[{compose}]", compose_args, eval_compose)

    substitution = {
        orig_compose: replacement(fused_compose)
        for orig_compose, fused_compose in compositions.items()
    }
    return term.replace(substitution), substitution


def collect_requires_bounds(
    term: Formula | Inequality | Expression | Function,
) -> tuple[Function, ...]:
    """
    Determine all :class:`Probability`, :class:`ExternalFunction`,
    and :class:`ExternalVariable` objects in the :class:`Formula`,
    :class:`Inequality`, :class:`Expression`, or :class:`Function`
    :code:`term`.
    """

    def is_prob_func_or_variable(term_):
        return isinstance(term_, Probability | ExternalFunction | ExternalVariable)

    return term.collect(is_prob_func_or_variable)
