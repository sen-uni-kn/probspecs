# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import sys
from enum import Enum, auto, unique
from itertools import cycle
from math import ceil
from typing import Callable, Literal, Generator
from time import time

import torch
import multiprocess as mp  # better multiprocessing using dill for serialization

from .bounds.auto_lirpa_params import AutoLiRPAParams
from .probability_distribution import ProbabilityDistribution
from .input_space import InputSpace
from .formula import (
    Formula,
    Probability,
    Function,
    Inequality,
    ExternalFunction,
    ExternalVariable,
)
from .trinary_logic import TrinaryLogic as TL
from .bounds.probability_bounds import probability_bounds
from .bounds.network_bounds import network_bounds
from .utils.formula_utils import (
    collect_requires_bounds,
    fuse_compositions,
    make_explicit,
)


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
    worker_devices: tuple[str | torch.device, ...] | None = None,
    parallel: bool = True,
    timeout: float | None = None,
    batch_size: int = 128,
    auto_lirpa_params: AutoLiRPAParams = AutoLiRPAParams(method="CROWN"),
    split_heuristic: Literal[
        "IBP", "CROWN", "longest-edge", "random", "prob-balanced"
    ] = "longest-edge",
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
    :param worker_devices: The devices (cpu, cuda:1, cuda:2, ...)
     to use for the different workers computing bounds.
     The length of this tuple determines the number of workers.
     If :code:`worker_devices` is :code:`None` (default), :code:`verify`
     uses all GPUs. If no GPUs are available, it uses half as many workers
     as there are (virtual) CPU cores.
    :param parallel: Whether to use parallel processing.
     If set to :code:`False`, bounds of terms are computed sequentially.
     If :code:`parallel=False`, only the first value of :code:`workers` is used.
    :param timeout: The timeout for verification.
     When verification exceeds this time budget, it is aborted.
    :param batch_size: The batch size to use for computing bounds.
    :param auto_lirpa_params: AutoLiRPA configuration for computing bounds
    :param split_heuristic: Split heuristic for computing bounds.
    :return: The status of verification and a verification witness.
     The verification witness is a set of bounds for functions in :code:`formula`
     that allows to prove/disprove :code:`formula`.
    """
    # Replace all given externals by ExplicitFunctions/Constants
    formula, make_explicit_subs = make_explicit(formula, **externals)
    # Replace compositions of functions by a single function computing the composition
    formula, compose_subs = fuse_compositions(formula)

    make_explicit_subs_reverse = {
        make_explicit_subs[term]: term for term in make_explicit_subs
    }
    compose_subs_reverse = {compose_subs[term]: term for term in compose_subs}

    def original_term(term):
        before_fuse_compose = term.replace(compose_subs_reverse)
        orig = before_fuse_compose.replace(make_explicit_subs_reverse)
        return orig

    requires_bounds = collect_requires_bounds(formula)
    bounded_terms_substitution = {
        term: ExternalVariable(f"?{term}?") for term in requires_bounds
    }
    # Use strings as keys because ExplicitFunction objects that are exchanged between
    # processes are not necessarily equal (due to the callables not being equal).
    # However, the string representations are equal.
    term_subs = {
        str(term): var.name for term, var in bounded_terms_substitution.items()
    }
    formula_skeleton = formula.replace(bounded_terms_substitution)

    if worker_devices is None:
        if torch.cuda.is_available():
            worker_devices = tuple(
                f"cuda:{i}" for i in range(torch.cuda.device_count())
            )
        else:
            worker_devices = ("cpu",) * (mp.cpu_count // 2)
    if not parallel:
        worker_devices = worker_devices[:1]
    worker_devices = [
        d if isinstance(d, torch.device) else torch.device(d) for d in worker_devices
    ]

    worker_processes = []
    bounds_queue = None
    try:
        # if there is just one worker, there is no need to start a separate
        # process
        if len(worker_devices) > 1:
            terms_per_worker = ceil(len(requires_bounds) / len(worker_devices))
            remaining_terms = list(requires_bounds)

            if any(d.type != "cpu" for d in worker_devices):
                if sys.platform.startswith("linux"):
                    mp_ctx = mp.get_context("forkserver")
                else:
                    mp_ctx = mp.get_context("spawn")
            else:
                mp_ctx = mp.get_context()
            bounds_queue = mp_ctx.SimpleQueue()
            for i, worker_device in enumerate(worker_devices):
                if len(remaining_terms) == 0:
                    break
                worker_terms = remaining_terms[:terms_per_worker]
                worker_terms = {str(term): term for term in worker_terms}
                remaining_terms = remaining_terms[terms_per_worker:]
                worker = mp_ctx.Process(
                    target=_compute_bounds_worker,
                    kwargs={
                        "target_terms": worker_terms,
                        "externals": externals,
                        "external_vars_domains": external_vars_domains,
                        "external_vars_distributions": external_vars_distributions,
                        "results_queue": bounds_queue,
                        "device": worker_device,
                        "batch_size": batch_size,
                        "auto_lirpa_params": auto_lirpa_params,
                        "split_heuristic": split_heuristic,
                    },
                )
                worker_processes.append(worker)
                worker.start()
        else:
            bounds_queue = _QueueEmulator(
                _compute_bounds_impl(
                    target_terms={str(term): term for term in requires_bounds},
                    externals=externals,
                    external_vars_domains=external_vars_domains,
                    external_vars_distributions=external_vars_distributions,
                    device=worker_devices[0],
                    batch_size=batch_size,
                    auto_lirpa_params=auto_lirpa_params,
                    split_heuristic=split_heuristic,
                )
            )

        def terminate_workers():
            for worker_ in worker_processes:
                worker_.terminate()

        start_time = time()

        def check_timeout():
            if timeout is not None and time() > start_time + timeout:
                terminate_workers()
                raise VerificationTimeout()

        def new_result():
            new_res = bounds_queue.get()
            if isinstance(new_res, Exception):
                terminate_workers()
                raise RuntimeError(
                    "Subprocess for computing bounds crashed."
                ) from new_res
            else:
                return new_res

        best_bounds = {}
        # fetch bounds until we have a bound for every term in requires_bounds
        while len(best_bounds) < len(requires_bounds):
            check_timeout()
            term_key, new_bounds = new_result()
            best_bounds[term_subs[term_key]] = new_bounds

        outcome = formula_skeleton.propagate_bounds(**best_bounds)
        while outcome == TL.UNKNOWN:
            check_timeout()
            term_key, new_bounds = new_result()
            best_bounds[term_subs[term_key]] = new_bounds
            outcome = formula_skeleton.propagate_bounds(**best_bounds)

        terminate_workers()
        best_bounds = {
            original_term(term): best_bounds[term_subs[str(term)]]
            for term in requires_bounds
        }
        return VerificationStatus.from_(outcome), best_bounds
    finally:
        if len(worker_processes) > 0:
            for worker in worker_processes:
                worker.terminate()
        if bounds_queue is not None:
            bounds_queue.close()


def _compute_bounds_impl(
    target_terms: dict[str, Function],
    externals: dict[str, Callable | torch.Tensor],
    external_vars_domains: dict[str, InputSpace],
    external_vars_distributions: dict[str, ProbabilityDistribution],
    device: torch.device | str,
    batch_size: int,
    auto_lirpa_params: AutoLiRPAParams,
    split_heuristic: Literal["IBP", "longest-edge"],
) -> Generator[tuple[str, tuple[torch.Tensor, torch.Tensor]], None, None]:
    """
    Computes bounds of one or multiple target terms and yields
    the new bounds sequentially, cycling indefinitely.
    The generator yields tuples of term keys (keys in :code:`target_terms`)
    and bounds.

    :param target_terms: A dictionary of term keys and terms (probabilities, networks)
     for which to compute bounds.
     The term keys are used to identify terms in the yielded bounds.
    :param externals: Values for external variables and function for evaluating
     the target terms.
    :param external_vars_domains: Bounds for external variables.
    :param external_vars_distributions: Probability distributions for external
     variables.
    :param device: Which device to use for computing bounds.
    """
    external_callables = {
        name: e for name, e in externals.items() if isinstance(e, Callable)
    }
    for external in external_callables.values():
        if isinstance(external, torch.nn.Module):
            external.to(device)

    def get_bounds_gen(term: Function):
        if isinstance(term, Probability):
            return probability_bounds(
                probability=term,
                networks=external_callables,
                variable_domains=external_vars_domains,
                variable_distributions=external_vars_distributions,
                batch_size=batch_size,
                auto_lirpa_params=auto_lirpa_params,
                split_heuristic=split_heuristic,
                device=device,
            )
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
            return network_bounds(
                network,
                input_bounds=arg_bounds,
                batch_size=batch_size,
                auto_lirpa_params=auto_lirpa_params,
                split_heuristic=split_heuristic,
            )

    bounds_gens = {
        term_key: get_bounds_gen(term) for term_key, term in target_terms.items()
    }
    for term_key, bounds_gen in cycle(bounds_gens.items()):
        new_bounds = next(bounds_gen)
        yield (term_key, new_bounds)


def _compute_bounds_worker(
    target_terms: dict[str, Function],
    externals: dict[str, Callable | torch.Tensor],
    external_vars_domains: dict[str, InputSpace],
    external_vars_distributions: dict[str, ProbabilityDistribution],
    results_queue: mp.Queue,
    device: torch.device | str,
    batch_size: int,
    auto_lirpa_params: AutoLiRPAParams,
    split_heuristic: Literal["IBP", "longest-edge"],
):
    """
    A worker for computing bounds on one or multiple target terms.
    Workers are intended to be run in parallel.
    However, one worker can compute bounds for several terms.
    These bounds are updated sequentially by the worker.

    The arguments are as for :code:_compute_bounds_impl`.

    :param results_queue: The :code:`multiprocessing.Queue` for posting results.
    """
    try:
        bounds_gen = _compute_bounds_impl(
            target_terms,
            externals,
            external_vars_domains,
            external_vars_distributions,
            device,
            batch_size,
            auto_lirpa_params,
            split_heuristic,
        )
        while True:
            new_result = next(bounds_gen)
            results_queue.put(new_result)
    except Exception as e:
        results_queue.put(e)
        sys.exit(1)


class _QueueEmulator:
    """
    Emulates a :code:`multiprocessing.SimpleQueue` object, while producing
    new values sequentially.
    """

    def __init__(self, results_generator: Generator):
        """
        Creates a new :class:`_QueueEmulator`.

        :param results_generator: The generator producing the values that
         :code:`_QueueEmulator` returns when :code:`get` is called.
        """
        self.__results_gen = results_generator

    def get(self):
        return next(self.__results_gen)

    def close(self):
        pass
