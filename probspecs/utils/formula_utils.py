# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Callable
from frozendict import frozendict

import torch

from ..formula import (
    Formula,
    Inequality,
    Expression,
    Function,
    Probability,
    ExternalFunction,
    ExternalVariable,
    Constant,
    Composition,
    ExplicitFunction,
    TERM_TYPES,
)


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


def make_explicit(
    term: TERM_TYPES,
    **values: torch.Tensor | float | Callable[..., torch.Tensor | float],
) -> TERM_TYPES:
    """
    Replaces (some) :class:`ExternalVariables` by :class:`Constants` and
    :class:`ExternalFunctions` by :class:`ExplicitFunctions`.

    Example:
    .. code-block::

      x = ExternalVariable("x")
      y = ExternalVariable("y")
      f = ExternalFunction("f", ("x", "y"))
      expr = f + 100.0

      def foo(x, y):
        return x * y / 2.0

      expr_new = make_explicit(expr, x=10.0, f=foo)
      # In expr_new, x is replaced by the Constant 10.0
      # and f is now an explicit function calling foo.

    When an external variable appears as the argument of a function, it's
    encoded in the function, such that it no longer needs to be supplied
    as an argument when evaluating the function or a term involving the function.
    However, for an :class:`ExternalFunction` :code:`f` with a variable
    :code:`x` as argument that's made explicit, this transformation
    introduces an :class:`ExplicitFunction` that no longer has :code:`x`
    as it's argument, but still the function :code:`f`.

    :param term: The term in which to replace :class:`ExternalVariables?
     and :class:`ExternalFunctions`.
    :param values: The values and callables to insert as :class:`Constants`
     and :class:`ExplicitFunctions` in :code:`term`.
     The variable or external function for which to insert a value or callable
     is specified by the argument name.
    :return: The updated :code:`term`.
    """

    def name(sub_term: TERM_TYPES):
        if isinstance(sub_term, ExternalVariable):
            return sub_term.name
        elif isinstance(sub_term, ExternalFunction):
            return sub_term.func_name
        else:
            return None

    def needs_replacement(sub_term: TERM_TYPES):
        if name(sub_term) in values:
            return True
        # also need to replace functions of which an argument is made explicit
        if isinstance(sub_term, ExternalFunction):
            return any(arg_name in values for arg_name in sub_term.arg_names)
        return False

    def replacement(
        sub_term: ExternalVariable | ExternalFunction,
    ):
        if isinstance(sub_term, ExternalVariable):
            return Constant(values[sub_term.name])
        elif isinstance(sub_term, ExternalFunction):
            new_args = tuple(
                arg_name for arg_name in sub_term.arg_names if arg_name not in values
            )
            func_name = sub_term.func_name
            if sub_term.func_name in values:
                func = values[sub_term.func_name]
            elif isinstance(sub_term, ExplicitFunction):
                func = sub_term.func
            else:
                func = None
                # Don't have the function, but have an argument
                # => require the function as argument of a new ExplicitFunction
                new_args = (sub_term.func_name,) + new_args
                # replace func_name by something like
                # f{x=0.35, y=15.3}
                sub_desc = ", ".join(
                    [
                        f"{arg_name}={values[arg_name]}"
                        for arg_name in sub_term.arg_names
                        if arg_name in values
                    ]
                )
                func_name = func_name + "{" + sub_desc + "}"

            def new_func(*args):
                if func is None:
                    func_ = args[0]
                else:
                    func_ = func
                args = dict(zip(new_args, args))
                args = (
                    args[arg_name] if arg_name in new_args else values[arg_name]
                    for arg_name in sub_term.arg_names  # old arg names
                )
                return func_(*args)

            return ExplicitFunction(func_name, new_args, new_func)
        else:
            raise NotImplementedError()

    to_replace = set(term.collect(needs_replacement))
    substitution = {sub: replacement(sub) for sub in to_replace}
    return term.replace(substitution)


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
    compositions = set(term.collect(lambda sub: isinstance(sub, Composition)))

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

        externals = compose.collect(is_external)  # may contain compose.func
        compose_args = sum((get_args(e) for e in externals), ())
        compose_args = set(compose_args)
        compose_args = tuple(arg for arg in compose_args if arg not in compose.args)

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
