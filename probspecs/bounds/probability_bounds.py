# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import typing
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from math import prod
from typing import Generator, Literal, Sequence, Callable, Final, Optional

import torch
from frozendict import frozendict
from torch import nn
from auto_LiRPA import BoundedModule
from ruamel.yaml import yaml_object

from .branch_store import BranchStore
from ..trinary_logic import TrinaryLogic
from ..formula import (
    ExternalFunction,
    Function,
    Probability,
    ExternalVariable,
    Constant,
    Inequality,
    Formula,
    ElementAccess,
)
from ..distributions.probability_distribution import ProbabilityDistribution
from ..input_space import (
    InputSpace,
    TensorInputSpace,
    TabularInputSpace,
    CombinedInputSpace,
)
from .utils import construct_bounded_tensor
from ..utils.formula_utils import (
    collect_requires_bounds,
    fuse_compositions,
    make_explicit,
)
from ..utils.config_container import ConfigContainer
from ..utils.names import Named
from ..utils.yaml import yaml

__all__ = [
    "SPLIT_HEURISTICS",
    "SPLIT_HEURISTICS_TYPE",
    "ProbabilityBounds",
    "BaseConfigScheduler",
    "WarmStartBounds",
    "MorePreciseBoundsOnPlateau",
    "SwitchToHeuristicsWithGuarantees",
    "SatBounds",
    "ProbabilityMass",
    "ScoreBranches",
    "Split",
    "SelectSplits",
]


class ProbabilityBounds(Named, ConfigContainer):
    """
    Computes a sequence of refined bounds on a probability.
    The :code:`bound` method produces a generator yielding lower and upper bounds
    on a :code:`Probability` term.
    The lower and upper bounds that this generator yields improve with successive
    yields, meaning that the lower bound increases while the upper bound decreases.
    In any case, the bounds are guaranteed (up to floating point error)
    to contain the actual probability.

    To refine the previously computed bounds, :code:`ProbabilityBounds` performs
    branch and bound with input splitting.
    There are various heuristics that the algorithm can use to select
    splits and branches.

    Split Heuristics:
     - :code:`"smart-branching"`: smart branching with IBP or CROWN.
       Split the dimension that leads to the best estimated bounds.
       The bounding method to use can be selected using the additional
       :code:`auto_lirpa_method` parameter (default is `"IBP"`).
       You can specify this parameter using the :code:`split_heuristic_params`
       argument.
       This heuristic has an additional parameter :code:`better_branch`
       that determines which branch that results from a split is selected
       to evaluate the split.
       By default (:code:`better_branch=True`), a split is evaluated using the
       bounds of the resulting branch with the better bounds.
       When :code:`better_branch=False`, instead the branch with the worse bounds
       is used.
       This is the strategy of BaBSR [BunelEtAl2020]_.
     - :code:`prob-weighted-smart-branching`: As smart branching, but computes
       the probability masses in each split and weights the bounds by the probability
       masses of the splits when selecting a branch.
     - :code:`"longest-edge"`: Split the dimension that has the largest diameter.
       By default, this strategy normalizes the diameters to the diameters of the
       initial variable bounds for each variable.
       This is to make sure that the algorithm does not concentrate on splitting one
       variable with a large initial diameter.
       You can turn off normalization by setting the split heuristic parameter
       :code:`normalize=False`.
     - :code:`"normalized-longest-edge"`: Split the dimension that has the largest
       diameter, where the diameter of each dimension is normalized by the initial
       diameter of the dimension.
       This variant of longest edge splitting is recommended for input spaces whose
       dimensions have different scales.
     - :code:`"random"`: Randomly selects a dimension to split.

    Branch Selection Heuristics:
     - :code:`"prob-mass"`: Split the branches with the largest probability mass in them.
     - :code:`"bounds"`: Selects the branch with the smallest lower or largest upper bounds
       (whichever is farther from zero).
       This broadly resembles the branch selection strategy from [BunelEtAl2020]_.
     - :code:`"prob-and-tight-bounds"`. Combines bounds and probabilities by multiplication.
       Broadly, the goal is to select the branch with the largest probability mass
       and the tightest bounds, searching for branches with high probability mass that
       are likely to be pruned soon.
       Concretely, we compute the distance between a lower and an upper bound,
       normalize this distance to [0.0, 1.0], subtract from one (so that
       tight bounds get large values) and multiply by the probability mass.
       We select the branches with the largest product of normalized bounds and
       probability masses.
     - :code:`"prob-and-log-bounds"`: As :code:`"prob-and-tight-bounds"`, but instead
       of normalizing the distances to [0.0, 1.0], we compute 1 / ln(distance + 2).
       This is then multiplied with the probability mass.
       The resulting product values are used to select branches.
       The idea here is to split based on probability mass when bounds are loose
       and take them into consideration when they are relatively tight.
     - :code:`"prob-and-loose-bounds"`: As :code:`"prob-and-tight-bounds"`, but
       prioritizes branches with high probability mass and loose bounds.

    .. [BunelEtAl2020] Rudy Bunel, Jingyue Lu, Ilker Turkaslan, Philip H. S. Torr,
        Pushmeet Kohli, M. Pawan Kumar: Branch and Bound for Piecewise Linear
        Neural Network Verification. J. Mach. Learn. Res. 21: 42:1-42:39 (2020)
    """

    def __init__(
        self,
        batch_size: int = 128,
        branch_selection_heuristic: "BRANCH_SELECTION_HEURISTIC_TYPE" = "prob-and-log-bounds",
        split_heuristic: "SPLIT_HEURISTICS_TYPE" = "smart-branching",
        split_heuristic_params: dict[
            Literal["auto_lirpa_method", "better_branch", "normalize"], bool
        ] = frozendict(),
        auto_lirpa_method: str = "CROWN",
        auto_lirpa_ops: dict = frozendict(),
        config_scheduler: Optional["ProbabilityBounds.ConfigSchedule"] = None,
        device: str | torch.device | None = None,
        random_seed: int = 0,
        log: bool = True,
    ):
        """
        Creates a new :code:`ProbabilityBounds` instance with the given configuration.

        :param batch_size: The number of branches to consider at a time.
        :param branch_selection_heuristic: Which heuristic to use for selecting branches
         to split.
        :param split_heuristic: Which heuristic to use for selecting dimensions to split.
        :param split_heuristic_params: Additional parameters of the split heuristic:
         - `"auto_lirpa_method"`: The auto_LiRPA method to use in smart branching.
           For example, `"IBP`" or "`CROWN`".
         - `"better_branch"`: When evaluating splits in `smart_branching`,
           calculate with the branch with the better bounds that results from the
           split or the worse?
         - `"normalize"`: Whether to normalize edge lengths to the initial variable bounds
           when using the `longest-edge` heuristic.
           This corresponds to normalizing the input variables to a common range.
           Enabling `"normalize"` is recommended for example, when working with both
           one-hot encoded categorical variables and numeric variables.
        :param auto_lirpa_method: The :code:`auto_LiRPA` bound propagation method
         to use for computing bounds.
        :param auto_lirpa_ops: :code:`auto_LiRPA` bound propagation options.
         More details in the :func:`auto_LiRPA.BoundedModule` documentation.
        :param config_scheduler: A config scheduler can update the configuration
         of this :code:`ProbabilityBounds` instance while it refines probability
         bounds.
        :param device: The device to compute on.
         If None, the tensors remain on the device they already reside on.
        :param random_seed: A random seed for branching and splitting heuristics that
         use randomness. The different heuristics add fixed constants to this seed,
         so that the heuristics do not have the same sequence of random numbers.
        :param log: Whether to print progress messages.
        """
        # Set the default split heuristic params if not given.
        split_heuristic_params = {
            "auto_lirpa_method": "IBP",
            "better_branch": True,
            "normalize": True,
        } | dict(split_heuristic_params)
        super().__init__(
            batch_size=batch_size,
            branch_selection_heuristic=branch_selection_heuristic,
            split_heuristic=split_heuristic,
            split_heuristic_params=split_heuristic_params,
            auto_lirpa_method=auto_lirpa_method,
            auto_lirpa_ops=auto_lirpa_ops,
            config_scheduler=config_scheduler,
            device=device,
            random_seed=random_seed,
            log=log,
        )

    config_keys = frozenset(
        {
            "batch_size",
            "branch_selection_heuristic",
            "split_heuristic",
            "split_heuristic_params",
            "auto_lirpa_method",
            "auto_lirpa_ops",
            "config_scheduler",
            "device",
            "random_seed",
            "log",
        }
    )

    def configure(self, **kwargs):
        """
        Configures this :code:`ProbabilityBounds` instance.
        The available configurations are documented at :code:`__init__`.
        """
        if (
            "branch_selection_heuristic" in kwargs
            and kwargs["branch_selection_heuristic"] not in BRANCH_SELECTION_HEURISTICS
        ):
            heuristic = kwargs["branch_selection_heuristic"]
            raise ValueError(
                f"Unknown branch selection heuristic: {heuristic}."
                f"Use one from {', '.join(BRANCH_SELECTION_HEURISTICS)}."
            )

        if (
            "split_heuristic" in kwargs
            and kwargs["split_heuristic"] not in SPLIT_HEURISTICS
        ):
            heuristic = kwargs["split_heuristic"]
            raise ValueError(
                f"Unknown split heuristic: {heuristic}."
                f"Use one from {', '.join(SPLIT_HEURISTICS)}."
            )
        super().configure(**kwargs)

    class ConfigScheduler(typing.Protocol):
        """
        A :code:`ConfigScheduler` reconfigures :code:`ProbabilityBounds` instances
        while bounds are refined.
        For example, a :code:`ConfigScheduler` may switch to more expensive bounding
        procedure when bounds stagnate or use a different split selection.

        :code:`ConfigSchedulers` are deep-copied when a :code:`ProbabilityBounds`
        instance delegates to new :code:`ProbabilityBounds` instances.
        """

        def setup(self, target: "ProbabilityBounds"):
            """
            Update the configuration of the :code:`target` :code:`ProbabilityBounds`
            instance.
            """
            ...

        def reconfigure(
            self,
            target: "ProbabilityBounds",
            iteration: int,
            prob_lb: float,
            prob_ub: float,
        ):
            """
            Update the configuration of the :code:`target` :code:`ProbabilityBounds`
            instance.

            :param target: The :code:`ProbabilityBounds` instance to reconfigure.
            :param iteration: The current iteration of :code:`target`.
            :param prob_lb: The current lower bound on the probability that
             :code:`target` computes bounds on.
            :param prob_ub: The current upper bound on the probability that
             :code:`target` computes bounds on.
            """
            ...

    def bound(
        self,
        probability: Probability,
        networks: dict[str, nn.Module],
        variable_domains: dict[str, InputSpace],
        variable_distributions: dict[str, ProbabilityDistribution],
    ) -> Generator[tuple[float, float], None, None]:
        """
        Computes a sequence of refined bounds on a probability.
        With each yield of this generator, the lower and upper bounds that it
        produces improve, meaning that the lower bound increases while the upper
        bound decreases.

        :param probability: The probability to compute bounds on.
        :param networks: The neural networks appearing as :class:`ExternalFunction`
         objects in :code:`probability`.
        :param variable_domains: :class:`InputSpace` objects for all external variables
         (:class:`ExternalVariable` objects and arguments of
         :class:`ExternalFunction` objects)
         appearing in :code:`probability`.
        :param variable_distributions: Probability distributions for all external variables
         appearing in :code:`probability` (see :code:`variable_domains`).
        :return: A generator that yields improving lower and upper bounds.
        """
        if probability.condition is not None:
            # use P(A|B) = P(A AND B) / P(B) to bound conditional probabilities.
            p_conjoined = Probability(probability.subject & probability.condition)
            p_condition = Probability(probability.condition)
            p_conjoined_bounds = ProbabilityBounds(
                **deepcopy(self.config)
            )._unconditional_probability_bounds(
                p_conjoined,
                networks,
                variable_domains,
                variable_distributions,
            )
            p_condition_bounds = ProbabilityBounds(
                **deepcopy(self.config)
            )._unconditional_probability_bounds(
                p_condition,
                networks,
                variable_domains,
                variable_distributions,
            )

            def conditional_bounds_gen():
                prob_formula = ExternalVariable("conj") / ExternalVariable("cond")
                for conj_bounds, cond_bounds in zip(
                    p_conjoined_bounds, p_condition_bounds
                ):
                    prob_bounds = prob_formula.propagate_bounds(
                        conj=conj_bounds, cond=cond_bounds
                    )
                    yield prob_bounds

            yield from conditional_bounds_gen()
        else:
            yield from self._unconditional_probability_bounds(
                probability, networks, variable_domains, variable_distributions
            )

    def _unconditional_probability_bounds(
        self,
        probability: Probability,
        networks: dict[str, nn.Module],
        variable_domains: dict[str, InputSpace],
        variable_distributions: dict[str, ProbabilityDistribution],
    ):
        """
        Computes successively refined bounds of a non-conditional probability.
        """
        full_input_space = CombinedInputSpace(variable_domains)

        # simplify the probability subject
        variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            var: domain.input_bounds for var, domain in variable_domains.items()
        }
        subj, _ = make_explicit(probability.subject, **networks)
        subj, _ = fuse_compositions(subj)
        subj, variable_bounds = apply_symbolic_bounds(subj, variable_bounds)

        # add batch dimensions to all bounds
        variable_bounds = {
            var: (lbs.unsqueeze(0).to(self.device), ubs.unsqueeze(0).to(self.device))
            for var, (lbs, ubs) in variable_bounds.items()
        }

        sat_bounds = SatBounds(subj, networks, variable_bounds, self)
        probability_mass = ProbabilityMass(variable_distributions)
        score_branches = ScoreBranches(self)
        select_splits = SelectSplits(
            full_input_space, self, sat_bounds, probability_mass
        )

        if self.config_scheduler is not None:
            self.config_scheduler.setup(self)

        branches = BranchStore(
            in_shape=full_input_space.input_shape,
            out_shape=(1,),  # a satisfaction score
            is_sat=(1,),  # trinary logic values for actual satisfaction
            probability_mass=(1,),
            device=self.device,
        )

        in_lbs = {var: lbs for var, (lbs, _) in variable_bounds.items()}
        in_ubs = {var: ubs for var, (_, ubs) in variable_bounds.items()}
        sat_lb, sat_ub, is_sat = sat_bounds(variable_bounds)
        prob_mass = probability_mass(variable_bounds)
        branches.append(
            in_lbs=full_input_space.combine(**in_lbs),
            in_ubs=full_input_space.combine(**in_ubs),
            out_lbs=sat_lb,
            out_ubs=sat_ub,
            is_sat=is_sat,
            probability_mass=prob_mass,
        )

        # prob_lb = probability mass in all regions where sat_lb > 0, which is a
        # sufficient condition for satisfaction.
        prob_lb = torch.tensor(0.0, dtype=prob_mass.dtype, device=self.device)
        # prob_ub = 1 - probability mass in all regions where sat_ub < 0, which is a
        # sufficient condition for violation.
        # Note that the total probability may be < 1.0
        prob_ub = torch.sum(branches.probability_mass)

        if self.log:
            print(
                f"[{self.name}]\n"
                f"Computing bounds on {probability}.\n"
                f"Config {self.config}\n" + ("-" * 100)
            )
        iteration = 0
        while True:
            # 0. Remove branches where subj is certainly satisfied or certainly violated.
            certainly_sat_mask = TrinaryLogic.is_true(branches.is_sat)
            certainly_viol_mask = TrinaryLogic.is_false(branches.is_sat)
            prob_lb += torch.sum(certainly_sat_mask * branches.probability_mass)
            prob_ub -= torch.sum(certainly_viol_mask * branches.probability_mass)
            branches.drop(certainly_sat_mask | certainly_viol_mask)

            if self.log:
                print(f"[{self.name}] New bounds: lb={prob_lb}, ub={prob_ub}")
            yield (prob_lb, prob_ub)

            if self.config_scheduler is not None:
                self.config_scheduler.reconfigure(self, iteration, prob_lb, prob_ub)

            # this is primarily for the case when apply_symbolic_bounds
            # removed all terms from subj
            # (for example, subj only contains bounds on variables)
            if torch.isclose(prob_lb, prob_ub, atol=0.0, rtol=1e-5):
                break

            if len(branches) == 0:
                raise RuntimeError(
                    f"No branches left, but bounds did not converge: "
                    f"{prob_lb} < {prob_ub}."
                )

            # 1. Select a batch of branches
            scores = score_branches(branches)
            branches.sort(scores, descending=True)
            selected_branches = branches.pop(self.batch_size)

            # 2. Select dimensions to split
            splits = select_splits(selected_branches)

            # 3. Split branches
            left_lbs, left_ubs = splits.left_branch
            right_lbs, right_ubs = splits.right_branch
            new_lbs = torch.vstack([left_lbs, right_lbs])
            new_ubs = torch.vstack([left_ubs, right_ubs])

            # 4. Update branches
            in_lbs = full_input_space.decompose(new_lbs)
            in_ubs = full_input_space.decompose(new_ubs)
            variable_bounds = {
                var: (in_lbs[var], in_ubs[var]) for var in variable_bounds
            }
            sat_lbs, sat_ubs, is_sat = sat_bounds(variable_bounds)
            prob_mass = probability_mass(variable_bounds)
            branches.append(
                in_lbs=new_lbs,
                in_ubs=new_ubs,
                out_lbs=sat_lbs.unsqueeze(1),
                out_ubs=sat_ubs.unsqueeze(1),
                is_sat=is_sat.unsqueeze(1),
                probability_mass=prob_mass,
            )

            iteration += 1

        while True:
            yield (prob_lb, prob_ub)


# MARK: config scheduling


@yaml_object(yaml)
class BaseConfigScheduler(ProbabilityBounds.ConfigScheduler, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__successor: Optional["BaseConfigScheduler"] = None
        self.__neighbour: Optional["BaseConfigScheduler"] = None

    def setup(self, target: "ProbabilityBounds"):
        if self.__neighbour is not None:
            self.__neighbour.setup(target)

    def reconfigure(
        self,
        target: "ProbabilityBounds",
        iteration: int,
        prob_lb: float,
        prob_ub: float,
    ):
        if self.__neighbour is not None:
            self.__neighbour.reconfigure(target, iteration, prob_lb, prob_ub)

    def _finish(self, target):
        if self.__successor is not None:
            target.configure(config_scheduler=self.__successor)
            self.__successor.setup(target)

    def chain(self, other: "BaseConfigScheduler"):
        if self.__successor is None:
            self.__successor = other
        else:
            self.__successor.parallel(other)

    def parallel(self, other: "BaseConfigScheduler"):
        if self.__neighbour is None:
            self.__neighbour = other
        else:
            self.__neighbour.parallel(other)

    # YAML serialization
    yaml_tag = "!BaseConfigScheduler"

    @property
    def state(self):
        """
        Return the state of this object for serialization.
        Need to be exactly your constructor arguments.
        """
        raise NotImplementedError()

    @classmethod
    def to_yaml(cls, representer, node):
        state = dict(node.state)
        if node.__successor is not None:
            state["successor"] = node.__successor
        if node.__neighbour is not None:
            state["neighbour"] = node.__neighbour
        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls, constructor, node):
        state = constructor.construct_mapping(node, deep=True)
        successor = neighbour = None
        if "successor" in state:
            successor = state["successor"]
            del state["successor"]
        if "neighbour" in state:
            neighbour = state["neighbour"]
            del state["neighbour"]
        obj = cls(**state)
        if successor is not None:
            obj.chain(successor)
        if neighbour is not None:
            obj.parallel(neighbour)
        return obj


@yaml_object(yaml)
class WarmStartBounds(ConfigContainer, BaseConfigScheduler):
    """
    Starts bound computation by refining the input space to improve
    network bounds.

    It sets:
     - `branch_selection_heuristic` to `bounds`
     - `auto_lirpa_method` to a user-set method, by default `IBP`.
     - `split_heuristic` to `smart-branching` with `IBP` and `better_branch=False`.
    """

    def __init__(self, iterations: int = 10, auto_lirpa_method: str = "IBP"):
        self.__initial_auto_lirpa_method = None
        self.__initial_branch_selection_heuristic = None
        self.__initial_split_heuristic = None
        self.__initial_split_heuristic_params = None
        self.__start_iter = None
        self.__done = False
        super().__init__(iterations=iterations, auto_lirpa_method=auto_lirpa_method)

    config_keys = frozenset({"iterations", "auto_lirpa_method"})

    def setup(self, target: "ProbabilityBounds"):
        self.__initial_auto_lirpa_method = target.auto_lirpa_method
        self.__initial_branch_selection_heuristic = target.branch_selection_heuristic
        self.__initial_split_heuristic = target.split_heuristic
        self.__initial_split_heuristic_params = target.split_heuristic_params
        target.configure(
            auto_lirpa_method=self.auto_lirpa_method,
            branch_selection_heuristic="bounds",
            split_heuristic="smart-branching",
            split_heuristic_params={"auto_lirpa_method": "IBP", "better_branch": False},
        )
        super().setup(target)

    def reconfigure(
        self,
        target: "ProbabilityBounds",
        iteration: int,
        prob_lb: float,
        prob_ub: float,
    ):
        try:
            if self.__done:  # Finished running
                return
            if self.__start_iter is None:
                print(f"[{target.name}] Warm-Starting Bounds.")
                self.__start_iter = iteration
            if iteration - self.__start_iter >= self.iterations:
                # End of warm start.
                print(
                    f"[{target.name}] Warm-Starting Bounds Finished. Resetting configuration."
                )
                target.auto_lirpa_method = self.__initial_auto_lirpa_method
                target.branch_selection_heuristic = (
                    self.__initial_branch_selection_heuristic
                )
                target.split_heuristic = self.__initial_split_heuristic
                target.split_heuristic_params = self.__initial_split_heuristic_params
                self.__done = True
                self._finish(target)
        finally:
            super().reconfigure(target, iteration, prob_lb, prob_ub)

    yaml_tag = "!WarmStartBounds"

    @property
    def state(self):
        return self.config


@yaml_object(yaml)
class MorePreciseBoundsOnPlateau(ConfigContainer, BaseConfigScheduler):
    """
    Uses successively more precise (and costly) `auto_lirpa_methods` when it
    detects the probability lower and upper bounds to plateau.

    By default, when detecting a plateau, it successively upgrades the
    `auto_lirpa_method` to
     - `CROWN`
     - `alpha-CROWN` with `lr=0.1` and `iterations=10`
     - `alpha-CROWN` with `lr=0.05` and `iterations=25`
     - `alpha-CROWN` with `lr=0.01` and `iterations=50`

    The scheduler detects a plateau if the bounds have not improved more than
    a certain threshold (1e-4 by default) for a certain number of iterations
    (`patience`, 10 by default).
    When a more precise bounding technique was selected, the scheduler lets it
    run for a certain number of iterations (`cooldown`, 10 by default) before
    tracking bounds again.
    Therefore, if `patience` and `cooldown` are both 10, the next better bounding
    technique will only be configured the earliest after 20 iterations after
    the last method update.
    """

    class Step(typing.NamedTuple):
        """
        One step in the method cascade of a :code:`MorePreciseBoundsOnPlateau`
        scheduler.
        Each step can have custom threshold, patience, and cooldown values.
        These values are for this step, that is, when the improvement in
        bounds lies below the threshold for `patience` iterations, the next
        step is configured, if there is one.
        """

        auto_lirpa_method: str
        auto_lirpa_ops: dict = {}
        threshold: float | None = None
        patience: int | None = None
        cooldown: int | None = None

        @classmethod
        def from_(cls, container: Sequence | typing.Mapping):
            if isinstance(container, Sequence):
                return cls(*container)
            else:
                return cls(
                    container["auto_lirpa_method"],
                    container["auto_lirpa_ops"],
                    container["threshold"],
                    container["patience"],
                    container["cooldown"],
                )

    def __init__(
        self,
        cascade: Sequence["MorePreciseBoundsOnPlateau.Step"] = (
            Step("IBP", threshold=1e-4),
            Step("CROWN-IBP", threshold=1e-4),
            Step("CROWN", threshold=1e-4),
        ),
        threshold=1e-4,
        patience=10,
        cooldown=10,
    ):
        """
        Create a new :code:`MorePreciseBoundsOnPlateau` config scheduler.

        :param cascade: A cascade of steps (four tuples of a :code:`auto_lirpa_method`,
         a :code:`threshold`, a :code:`patience` and a :code:`cooldown`).
         The threshold, patience, and cooldown may be :code:`None`, in which case
         the default values from the remaining arguments of this method are used.
        :param threshold: The default threshold for advancing to the next step in
         the cascade.
        :param patience: The default patience for advancing to the next step in the
         cascade.
        :param cooldown: The default cooldown iterations for one step in the cascade.
        """
        cascade = tuple(
            MorePreciseBoundsOnPlateau.Step.from_(container) for container in cascade
        )
        cascade = tuple(
            MorePreciseBoundsOnPlateau.Step(
                method,
                ops,
                threshold if t is None else t,
                patience if p is None else p,
                cooldown if c is None else c,
            )
            for method, ops, t, p, c in cascade
        )
        super().__init__(cascade=cascade)
        self.__step = 0
        self.__cooldown_counter = 0
        self.__lb_history = []  # maximal length: patience
        self.__ub_history = []  # maximal length: patience

    config_keys = frozenset({"cascade"})

    def _configure_step(self, step: int, target: "ProbabilityBounds") -> str:
        method, ops, _, _, _ = self.cascade[step]
        # keep all ops that we don't change
        new_ops = dict(target.auto_lirpa_ops) | ops
        target.configure(auto_lirpa_method=method, auto_lirpa_ops=new_ops)
        self.__lb_history = []
        self.__ub_history = []
        return f"Now using auto_lirpa_method={method}, auto_lirpa_ops={new_ops}."

    def setup(self, target: "ProbabilityBounds"):
        message = self._configure_step(0, target)
        print(f"[{target.name}] MorePreciseBoundsOnPlateau scheduler set up. {message}")
        super().setup(target)

    def reconfigure(
        self,
        target: "ProbabilityBounds",
        iteration: int,
        prob_lb: float,
        prob_ub: float,
    ):
        if self.__step == len(self.cascade) - 1:
            return
        try:
            _, _, threshold, patience, cooldown = self.cascade[self.__step]
            if self.__cooldown_counter < cooldown:
                self.__cooldown_counter += 1
                return

            self.__lb_history.append(prob_lb)
            self.__ub_history.append(prob_ub)
            self.__lb_history = self.__lb_history[-patience:]
            self.__ub_history = self.__ub_history[-patience:]

            if len(self.__lb_history) == patience:
                improvement_lb = prob_lb - self.__lb_history[0]
                improvement_ub = self.__ub_history[0] - prob_ub
                if max(improvement_lb, improvement_ub) > threshold:
                    return

                self.__step += 1
                message = self._configure_step(self.__step, target)
                print(
                    f"[{target.name}] Bounds Plateauing. Updating bounding method. "
                    f"{message}"
                )
        finally:
            super().reconfigure(target, iteration, prob_lb, prob_ub)

    yaml_tag = "!MorePreciseBoundsOnPlateau"

    @property
    def state(self):
        return {"cascade": [step._asdict() for step in self.cascade]}


@yaml_object(yaml)
class SwitchToHeuristicsWithGuarantees(ConfigContainer, BaseConfigScheduler):
    """
    Switches to a configuration that affords convergence guarantees
    after a certain number of iterations.

    Concretely, it sets:
     - `branch_selection_heuristics` to `prob-mass`
     - 'split_heuristic` to `longest-edge`
     - `auto_lirpa_method` to `IBP`.

    The purpose of this scheduler is to warm-start a configuration that
    affords convergence guarantees with a set of heuristics that is
    practically advantageous.
    This maintains the convergence guarantee.
    """

    def __init__(self, switch_after: int = 100):
        super().__init__(switch_after=switch_after)

    config_keys = frozenset({"switch_after"})

    def reconfigure(
        self,
        target: "ProbabilityBounds",
        iteration: int,
        prob_lb: float,
        prob_ub: float,
    ):
        if iteration == self.switch_after:
            print(
                f"[{target.name}] Switching to Heuristics with "
                f"Convergence Guarantees."
            )
            target.branch_selection_heuristic = "prob-mass"
            target.split_heuristic = "longest-edge"
            target.auto_lirpa_method = "IBP"
        super().reconfigure(target, iteration, prob_lb, prob_ub)

    yaml_tag = "!SwitchToHeuristicsWithGuarantees"

    @property
    def state(self):
        return self.config


# MARK: preprocessing


def apply_symbolic_bounds(
    formula: Formula | Inequality,
    variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[Formula, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """
    If :code:`formula` has the structure
    :code:`x >= y AND ...`,
    where :code:`x` is an :class:`ExternalVariable` and
    :code:`y` is a :class:`Constant`,
    the term :code:`x >= y` can be simplified when the values
    for which the formula is evaluated are bounded.
    Concretely, when we have bounds on the external variable :code:`x`,
    we can removed the term :code:`x >= y` from the formula
    applying :code:`x >= y` directly to the bounds on :code:`x`.
    This function performs this simplification.
    This function can also apply multiple simple inequalities, including
    inequalities like :code:`y <= x`.

    :param formula: The :class:`Formula` to simplify.
     For convenience, this function als accepts :class:`Inequality` objects.
     When an inequality can be simplified, the formula that this function returns
     is equivalent to :code:`True`.
    :param variable_bounds: Bounds on all external variables appearing in
     :code:`formula`.
    :return: The simplified formula and the new variable bounds
     where the term removed from the formula were applied directly.
    """

    def is_simple(term_: Formula | Inequality) -> bool:
        """
        Simple terms: x >= c, x <= c, c >= x, c <= x,
        x[...] >= c, x[...] <= c, c >= x[...], c <= x[...]
        where c is a Constant and x is an ExternalVariable.
        """
        if isinstance(term_, Inequality) and term_.op in (
            Inequality.Operator.LESS_EQUAL,
            Inequality.Operator.GREATER_EQUAL,
        ):
            if isinstance(term_.lhs, Constant):
                const_ = term_.lhs
                other_ = term_.rhs
            elif isinstance(term_.rhs, Constant):
                const_ = term_.rhs
                other_ = term_.lhs
            else:
                const_ = other_ = None

            if const_ is not None:
                return isinstance(other_, ExternalVariable) or (
                    isinstance(other_, ElementAccess)
                    and isinstance(other_.source, ExternalVariable)
                )
        return False

    if isinstance(formula, Formula) and formula.op is Formula.Operator.AND:
        simple_terms = []
        remaining_terms = []
        # This AND may contain nested ANDs
        top_level_conjuncts = [formula]
        while len(top_level_conjuncts) > 0:
            conj = top_level_conjuncts.pop()
            for term in conj.operands:
                if is_simple(term):
                    simple_terms.append(term)
                elif isinstance(term, Formula) and term.op is Formula.Operator.AND:
                    top_level_conjuncts.append(term)
                else:
                    remaining_terms.append(term)
    elif is_simple(formula):
        simple_terms = [formula]
        remaining_terms = []
    else:
        return formula, variable_bounds

    new_bounds = dict(variable_bounds)
    for ineq in simple_terms:
        const = ineq.lhs if isinstance(ineq.lhs, Constant) else ineq.rhs
        other = ineq.rhs if isinstance(ineq.lhs, Constant) else ineq.lhs

        if isinstance(other, ExternalVariable):
            var = other
            lb, ub = new_bounds[var.name]
            if ineq.op is Inequality.Operator.LESS_EQUAL:
                ub = torch.clamp(ub, max=const.val)
            else:  # var >= const
                lb = torch.clamp(lb, min=const.val)
            new_bounds[var.name] = (lb, ub)
        elif isinstance(other, ElementAccess):
            var = other.source
            lb, ub = new_bounds[var.name]
            # deal with missing batch dimensions in bounds
            item = other.target_item[1:]
            lb_item = lb[item]
            ub_item = ub[item]
            if ineq.op is Inequality.Operator.LESS_EQUAL:
                ub_item = torch.clamp(ub_item, max=const.val)
            else:  # var >= const
                lb_item = torch.clamp(lb_item, min=const.val)
            lb[item] = lb_item
            ub[item] = ub_item
            new_bounds[var.name] = (lb, ub)
        else:
            raise NotImplementedError()

    remaining_terms = Formula(Formula.Operator.AND, tuple(remaining_terms))
    return remaining_terms, new_bounds


# MARK: computing bounds and probability masses


class SatBounds:
    """
    Computes bounds on the satisfaction function of a :class:`Formula` (the `subject`).
    """

    class Config(typing.Protocol):
        @property
        def auto_lirpa_method(self) -> str: ...

        @property
        def auto_lirpa_ops(self) -> dict: ...

        @property
        def device(self) -> torch.device: ...

    def __init__(
        self,
        subject: Formula,
        networks: dict[str, nn.Module],
        variable_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
        config: "SatBounds.Config",
    ):
        self._config = config

        self._requires_bounds = set(collect_requires_bounds(subject))
        self._check_for_unsupported_probabilities(subject)

        self._networks = {
            external.func_name: BoundedModule(
                external.get_function(**networks),
                variable_bounds[external.arg_names[0]][0],  # lower bound of arg
                config.auto_lirpa_ops,
                config.device,
            )
            for external in self._requires_bounds
            if isinstance(external, ExternalFunction)
        }

        self._bounded_terms_substitution = {
            term: ExternalVariable(f"?{term}?") for term in self._requires_bounds
        }
        self._subs_names = {
            term: var.name for term, var in self._bounded_terms_substitution.items()
        }
        self._subj_skeleton = subject.replace(self._bounded_terms_substitution)
        self._subj_skeleton_sat_fn = self._subj_skeleton.satisfaction_function

    def _check_for_unsupported_probabilities(self, subject):
        # make sure probability doesn't contain nested probabilities.
        for term in self._requires_bounds:
            if isinstance(term, Probability):
                raise ValueError(
                    "Computing bounds on probabilities with"
                    "nested probabilities is unsupported. "
                    f"Found probability {term} in {subject}."
                )

        # auto_LiRPA: no networks with multiple inputs
        for external in self._requires_bounds:
            if isinstance(external, ExternalFunction):
                if len(external.arg_names) > 1:
                    raise ValueError(
                        f"ExternalFunctions with multiple inputs are unsupported "
                        f"by probability_bounds. "
                        f"Found function with multiple arguments: {external}."
                    )

    def __call__(
        self,
        var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
        auto_lirpa_method: str | None = None,
        auto_lirpa_ops: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sat_bounds(var_bounds, auto_lirpa_method, auto_lirpa_ops)

    @torch.no_grad()
    def sat_bounds(
        self,
        var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
        auto_lirpa_method: str | None = None,
        auto_lirpa_ops: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns lower and upper bounds on the satisfaction function
        (first two tensors) and a tensor of trinary truth values indicating
        whether :code:`subj_skeleton` is satisfied.

        The additional trinary logic values are useful since a satisfaction function
        value of 0.0 is inconclusive.

        :param var_bounds: The bounds of the external variables in the formula of this
         :code:`SatBounds` instance.
        :param auto_lirpa_method: Allows overwriting the default `auto_lirpa_method`
         that is stored in `config` (initializer argument).
        """
        intermediate_bounds = {
            self._subs_names[term]: self._propagate_bounds(
                term, var_bounds, auto_lirpa_method, auto_lirpa_ops
            )
            for term in self._requires_bounds
        }
        sat_lb, sat_ub = self._subj_skeleton_sat_fn.propagate_bounds(
            **intermediate_bounds
        )
        is_sat = self._subj_skeleton.propagate_bounds(**intermediate_bounds)
        if not isinstance(is_sat, torch.Tensor):
            is_sat = torch.tensor([is_sat])
        sat_lb = sat_lb.to(self._config.device)
        sat_ub = sat_ub.to(self._config.device)
        is_sat = is_sat.to(self._config.device)
        return sat_lb, sat_ub, is_sat

    def _propagate_bounds(
        self,
        term_: Function,
        var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
        auto_lirpa_method: str | None = None,
        auto_lirpa_ops: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(term_, ExternalVariable):
            return term_.propagate_bounds(**var_bounds)
        elif isinstance(term_, ExternalFunction):
            method = self._config.auto_lirpa_method
            ops = self._config.auto_lirpa_ops
            if auto_lirpa_method is not None:
                method = auto_lirpa_method
            if auto_lirpa_ops is not None:
                ops = auto_lirpa_ops

            network = self._networks[term_.func_name]
            network.set_bound_opts(ops)
            bounded_tensor = construct_bounded_tensor(*var_bounds[term_.arg_names[0]])
            network(bounded_tensor)
            return network.compute_bounds(x=(bounded_tensor,), method=method)
        else:
            raise NotImplementedError()


class ProbabilityMass:
    """
    Computes probability masses.
    """

    def __init__(self, variable_distributions: dict[str, ProbabilityDistribution]):
        self._variable_distributions = variable_distributions

    def __call__(
        self, var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        return self.probability_mass(var_bounds)

    @torch.no_grad()
    def probability_mass(
        self, var_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        probs = (
            self._variable_distributions[var].probability(bounds)
            for var, bounds in var_bounds.items()
        )
        return prod(probs).reshape(-1, 1)


# MARK: score branches

BRANCH_SELECTION_HEURISTIC_TYPE: Final = Literal[
    "prob-mass",
    "bounds",
    "prob-and-log-bounds",
    "prob-and-lin-prune",
    "prob-and-tight-bounds",
    "prob-and-loose-bounds",
    "random",
]
BRANCH_SELECTION_HEURISTICS: Final[tuple[BRANCH_SELECTION_HEURISTIC_TYPE, ...]] = (
    typing.get_args(BRANCH_SELECTION_HEURISTIC_TYPE)
)


class ScoreBranches:
    """
    Scores branches for selecting the most promising branches to split.
    Branches with a higher score are more promising.
    """

    class Config(typing.Protocol):
        @property
        def branch_selection_heuristic(self) -> BRANCH_SELECTION_HEURISTIC_TYPE: ...

        @property
        def device(self) -> torch.device:
            ...

        @property
        def random_seed(self) -> int:
            ...

    def __init__(self, config: "ScoreBranches.Config"):
        self._config = config

        # For random branch scoring
        self._rng = torch.Generator(device=config.device)
        self._rng.manual_seed(config.random_seed + 250587974283011)

    def __call__(self, branches) -> torch.Tensor:
        return self.score(branches)

    def score(self, branches: BranchStore) -> torch.Tensor:
        match self._config.branch_selection_heuristic:
            case "prob-mass":
                return self.score_probability_mass(branches)
            case "bounds":
                return self.score_bounds(branches)
            case "prob-and-log-bounds":
                return self.score_prob_and_bounds(branches, "log-decay")
            case "prob-and-lin-prune":
                return self.score_prob_and_prune(branches, "linear")
            case "prob-and-tight-bounds":
                return self.score_prob_and_bounds(
                    branches, "linear", prefer_tight_bounds=True
                )
            case "prob-and-loose-bounds":
                return self.score_prob_and_bounds(
                    branches, "linear", prefer_tight_bounds=False
                )
            case "random":
                return self.score_random(branches)
            case _:
                raise NotImplementedError()

    def score_probability_mass(self, branches: BranchStore) -> torch.Tensor:
        return branches.probability_mass.flatten()

    def score_bounds(self, branches: BranchStore) -> torch.Tensor:
        out_lbs = branches.out_lbs
        out_ubs = branches.out_ubs
        # select the branches with the loosest bounds.
        return torch.maximum(out_ubs, -out_lbs).flatten()

    def score_prob_and_bounds(
        self,
        branches: BranchStore,
        transform_bounds: Literal["log-decay", "linear"],
        prefer_tight_bounds: bool = True,
    ) -> torch.Tensor:
        """
        Selects branches with relative tight bounds and have large
        probability mass.
        Selecting branches with tight bounds is preferable, as the bounding
        procedure might be able to prune them soon.
        """
        out_lbs = branches.out_lbs
        out_ubs = branches.out_ubs
        prob_mass = branches.probability_mass
        # we use the bounds range as an indicator for tightness.
        bounds_range = out_ubs - out_lbs
        # For unbounded input spaces, bounds may be `nan`.
        # Since we don't know the actual bounds, we use the mean range for these.
        bounds_range = torch.where(
            torch.isnan(bounds_range), torch.nanmean(bounds_range), bounds_range
        )
        # if all ranges are nan, bounds_range is still nan.
        bounds_range = bounds_range.nan_to_num(
            nan=0.0
        )  # => falls back to prob_mass only
        match transform_bounds:
            case "log-decay":
                # decay: normalize to [0, 1], large ranges get lower priority over probabilities
                bounds_decayed = 1 / torch.log(bounds_range + torch.e)
                if not prefer_tight_bounds:
                    bounds_decayed = 1 - bounds_decayed
            case "linear":
                # not really a decay.
                min_range = torch.amin(bounds_range)
                max_range = torch.amax(bounds_range)
                # map max_range to 1.0 and min_range to 0.0
                bounds_decayed = (bounds_range - min_range) / (max_range - min_range)
                # flip direction: make min range 1.0 (since min_range is better)
                if prefer_tight_bounds:
                    bounds_decayed = 1 - bounds_decayed
            case _:
                raise NotImplementedError()
        return torch.mul(prob_mass, bounds_decayed).flatten()

    def score_prob_and_prune(
        self, branches: BranchStore, transform_bounds: Literal["linear"]
    ):
        """
        Select branches with high probability mass and either the lower output bound
        or the upper output bound close to zero.

        The motivation for this is to greedily select the branches that will likely
        contribute the most to the computed probability bounds in the next iteration.
        To estimate this, we approximate the likelyhood of a branch to be pruned
        by its lower or upper bound being close to zero.
        By normalizing the absolute value of the bounds and multiplying the normalized
        value with the probability mass in the branch, we obtain the desired estimate.
        """
        out_lbs = branches.out_lbs
        out_ubs = branches.out_ubs
        prob_mass = branches.probability_mass

        match transform_bounds:
            case "linear":
                # Normalize bounds to [0, 1], where 1.0 is the largest absolute bounds
                # and 0.0 is the actual zero.
                # For unbounded input spaces, bounds may be `nan` or `inf`. Ignore now.
                # We later give them the average tightness value.
                lbs_abs_max = out_lbs.nan_to_num(0.0, 0.0, 0.0).abs().max()
                ubs_abs_max = out_ubs.nan_to_num(0.0, 0.0, 0.0).abs().max()
                bounds_abs_max = torch.maximum(lbs_abs_max, ubs_abs_max)
                if torch.isclose(bounds_abs_max, 0.0):
                    # this happens when all bounds are nan or inf
                    # In this case, we fall back to just prob-mass
                    bounds_score = torch.ones_like(out_lbs)
                else:
                    bounds_score = torch.minimum(-out_lbs, out_ubs) / bounds_abs_max
                    # see SelectSplits -> smart_branching
                    bounds_score = torch.round(bounds_score, decimals=4)
                    # we prefer bounds close to zero (likely to be pruned)
                    bounds_score = 1 - bounds_score
            case _:
                raise NotImplementedError()
        return torch.mul(prob_mass, bounds_score).flatten()

    def score_random(self, branches: BranchStore):
        """
        Assign a random score to all branches.
        """
        return torch.rand(len(branches), generator=self._rng)


# MARK: splitting


SPLIT_HEURISTICS_TYPE: Final = Literal[
    "smart-branching",
    "prob-weighted-smart-branching",
    "longest-edge",
    "random",
]
SPLIT_HEURISTICS: Final[tuple[SPLIT_HEURISTICS_TYPE, ...]] = typing.get_args(
    SPLIT_HEURISTICS_TYPE
)


@dataclass
class Split:
    """
    The two branches of a split. Each branch is represented by
    a lower and an upper bound.
    """

    left_branch: tuple[torch.Tensor, torch.Tensor]
    right_branch: tuple[torch.Tensor, torch.Tensor]

    @staticmethod
    def stack(splits: Sequence["Split"], dim=0) -> "Split":
        """
        Stacks (:func:`torch.stack`) the bounds tensors of
        several :class:`Split` instances.
        Produces a batched :class:`Split` representing several splits.

        :param splits: The :class:`Split` instances to stack.
        :param dim: See :code:`torch.stack`.
        :return: A batched :class:`Split`.
        """
        kwargs = {}
        for branch in ("left_branch", "right_branch"):
            lbs = []
            ubs = []
            for s in splits:
                lb, ub = getattr(s, branch)
                lbs.append(lb)
                ubs.append(ub)
            kwargs[branch] = (torch.stack(lbs, dim), torch.stack(ubs, dim))
        return Split(**kwargs)

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Split":
        """
        Applies a function :code:`fn` to the `mask` and `values` tensors of
        the :class:`TensorUpdate` instances underlying this :class:`Split`.
        :param fn: The function to apply to the underlying `mask` and `values` tensors.
        :return: A :class:`Split` with :code:`fn` applied to the `mask` and `values`
         tensors.
        """
        kwargs = {}
        for branch in ("left_branch", "right_branch"):
            lb, ub = getattr(self, branch)
            kwargs[branch] = (fn(lb), fn(ub))
        return Split(**kwargs)

    def select(self, selected_splits: torch.Tensor) -> "Split":
        """
        Select a split for each batch element.

        :param selected_splits: A 1d tensor containing the index of one
         split to select for each batch element.
        :return: A :class:`Split` object containing the selected splits.
        """
        selected_splits = selected_splits.reshape(1, -1, 1)

        def take_selected(t: torch.Tensor):
            return t.take_along_dim(selected_splits, dim=0).squeeze(0)

        return self.map(take_selected)


class SelectSplits:
    """Select which dimensions to split for a batch of branches."""

    class Config(typing.Protocol):
        @property
        def split_heuristic(self) -> SPLIT_HEURISTICS_TYPE: ...

        @property
        def split_heuristic_params(self) -> dict[str, typing.Any]: ...

        @property
        def device(self) -> torch.device: ...

        @property
        def random_seed(self) -> int:
            ...

    def __init__(
        self,
        input_space: CombinedInputSpace,
        config: "SelectSplits.Config",
        sat_bounds: SatBounds,
        prob_mass: ProbabilityMass,
    ):
        self._input_space = input_space
        self._config = config
        self._sat_bounds = sat_bounds
        self._prob_mass = prob_mass

        # Some split heuristics resolve ties using randomness to avoid that the
        # first dimensions are split more frequently than later dimensions.
        self._rng = torch.Generator(device=config.device)
        self._rng.manual_seed(config.random_seed + 9876543210)

    def __call__(self, branches) -> Split:
        return self.select_splits(branches)

    def select_splits(self, branches: BranchStore) -> Split:
        match self._config.split_heuristic:
            case "smart-branching":
                return self.smart_branching(branches)
            case "prob-weighted-smart-branching":
                return self.smart_branching(branches, prob_weighted=True)
            case "longest-edge":
                return self.split_longest_edge(branches)
            case "random":
                return self.split_random(branches)
            case _:
                raise NotImplementedError()

    @torch.no_grad()
    def smart_branching(self, branches: BranchStore, prob_weighted=False) -> Split:
        """
        Select input dimensions to split.
        :code:`smart_branching` selects the dimension (for each batch element)
        that leads to the largest improvement in the estimated lower or upper bound
        of the satisfaction function when split.
        The improvements in lower and upper bounds are estimated using the
        bounding algorithm :code:`method`.
        You should select a cheap bounding algorithm, such as IBP for this task.
        This split strategy is an adaption of BaBSR from [BunelEtAl2020]_.

        The algorithm to compute bounds can be selected using the
        :code:`auto_lirpa_method` split heuristic parameter.

        Each split is evaluated either by selecting the branch resulting from the split
        that has the better or worse bounds.
        Which is selected is determined by the :code:`better_branch`
        split heuristic parameter.
        By default, the better bounds are used, but using the worse bounds may lead to
        a more balanced branching tree.

        .. [BunelEtAl2020] Rudy Bunel, Jingyue Lu, Ilker Turkaslan, Philip H. S. Torr,
            Pushmeet Kohli, M. Pawan Kumar: Branch and Bound for Piecewise Linear
            Neural Network Verification. J. Mach. Learn. Res. 21: 42:1-42:39 (2020)

        :param branches: The branches for which to determine the dimensions to split.
        :param prob_weighted: Whether consider probabilities when selecting splits.
        :return: The splits to perform.
        """
        splits, is_invalid = self.propose_splits(branches)
        left_bounds, right_bounds, num_splits, batch_size = self._get_branch_bounds(
            splits
        )

        method = self._config.split_heuristic_params["auto_lirpa_method"]
        # We already use very large batch sizes here since left_lbs contains
        # num input dims * original batch size many elements.
        # To balance memory requirements better, it makes sense to perform ibp/crown
        # twice here instead of one call with an extremely large batch size.
        left_sat_lbs, left_sat_ubs, _ = self._sat_bounds(
            left_bounds, auto_lirpa_method=method
        )
        right_sat_lbs, right_sat_ubs, _ = self._sat_bounds(
            right_bounds, auto_lirpa_method=method
        )

        # For unbounded input spaces, bounds may be `nan`. Since nan hampers with
        # taking the maximum and minimum, we need to replace nan by a proper value.
        # Since the selection will not depend on the bounds in this case
        # (we always want to select an unbounded dimension), we just replace nan by 0.0.
        left_sat_lbs = left_sat_lbs.nan_to_num()
        left_sat_ubs = left_sat_ubs.nan_to_num()
        right_sat_lbs = right_sat_lbs.nan_to_num()
        right_sat_ubs = right_sat_ubs.nan_to_num()

        # recreate (split dims, batch) shape (individual sat bounds are scalars)
        left_sat_lbs = left_sat_lbs.reshape(num_splits, batch_size)
        left_sat_ubs = left_sat_ubs.reshape(num_splits, batch_size)
        right_sat_lbs = right_sat_lbs.reshape(num_splits, batch_size)
        right_sat_ubs = right_sat_ubs.reshape(num_splits, batch_size)

        if prob_weighted:
            left_prob_mass = self._prob_mass(left_bounds)
            right_prob_mass = self._prob_mass(right_bounds)
            left_prob_mass = left_prob_mass.reshape(num_splits, batch_size)
            right_prob_mass = right_prob_mass.reshape(num_splits, batch_size)
            # We want lower bounds to be large, upper bounds to be small,
            # and probabilities of pruned branches should be large.
            # Shift bounds, so that we don't multiply the probability masses by zero.
            max_lbs = max(torch.amax(left_sat_lbs), torch.amax(right_sat_lbs))
            min_ubs = min(torch.amin(left_sat_ubs), torch.amin(right_sat_ubs))
            left_sat_lbs = left_prob_mass * (left_sat_lbs - max_lbs - 1.0)
            left_sat_ubs = (1 - left_prob_mass) * (left_sat_ubs + min_ubs + 1.0)
            right_sat_lbs = right_prob_mass * (right_sat_lbs - max_lbs - 1.0)
            right_sat_ubs = (1 - right_prob_mass) * (right_sat_ubs + min_ubs + 1.0)

        # Split selection strategy:
        # Select the split which makes the lower/upper bound on the branch
        # with the smaller/larger lower/upper bound largest/smallest.
        # Branches which make the lower bound positive or the upper bound negative
        # are the best splits, as they allow for pruning the branch.
        #
        # Adapted from: BaBSR (Rudy Bunel, Jingyue Lu, Ilker Turkaslan, Philip H. S. Torr,
        # Pushmeet Kohli, M. Pawan Kumar: Branch and Bound for Piecewise Linear
        # Neural Network Verification. J. Mach. Learn. Res. 21: 42:1-42:39 (2020))
        if self._config.split_heuristic_params["better_branch"]:
            score_sat_lbs = torch.maximum(left_sat_lbs, right_sat_lbs)
            score_sat_ubs = torch.minimum(left_sat_ubs, right_sat_ubs)
        else:
            score_sat_lbs = torch.minimum(left_sat_lbs, right_sat_lbs)
            score_sat_ubs = torch.maximum(left_sat_ubs, right_sat_ubs)

        select_score = torch.maximum(score_sat_lbs, -score_sat_ubs)
        # At times splitting bounds that actually influence the output can lead
        # to slightly worse bounds than splitting an input that does not influence
        # the output due to floating point error.
        # To avoid this, we round the scores.
        select_score = torch.round(select_score, decimals=4)

        # Ensure unbounded dimensions are always selected first so that we can compute
        # sensible bounds next time
        unbounded_dim = ~branches.in_lbs.isfinite() | ~branches.in_ubs.isfinite()
        unbounded_dim = unbounded_dim.permute(1, 0)
        select_score.masked_fill_(unbounded_dim, torch.inf)

        # give invalid splits maximally bad sat scores so that they aren't selected.
        select_score.masked_fill_(is_invalid, -torch.inf)

        # resolve ties in select_scores randomly to avoid always splitting on the
        # early dimensions when bounds are very loose
        permute = torch.randperm(
            num_splits, generator=self._rng, device=self._rng.device
        )
        select_score = select_score[permute, :]
        split_dims = torch.argmax(select_score, dim=0)
        split_dims = permute[split_dims]
        return splits.select(split_dims)

    def split_longest_edge(self, branches: BranchStore) -> Split:
        """
        Select input dimensions to split.
        :code:`split_longest_edge` selects the dimension (for each batch element)
        that has the largest distance between lower and upper bound (the longest edge).

        Optionally allows to normalize the edge lengths to the initial edge lengths
        as specified in the :code:`input_space` (initializer argument).

        :param branches: The branches for which to determine the dimensions to split.
        :return: The splits to perform.
        """
        # 1st dim of tensors in splits corresponds to the dimension that is split
        splits, is_invalid = self.propose_splits(branches)

        edge_len = branches.in_ubs - branches.in_lbs
        if self._config.split_heuristic_params["normalize"]:
            initial_lb, initial_ub = self._input_space.input_bounds
            initial_lb, initial_ub = initial_lb.to(edge_len), initial_ub.to(edge_len)
            edge_len = edge_len / (initial_ub - initial_lb)
        edge_len.masked_fill_(is_invalid.T, -torch.inf)
        split_dims = torch.argmax(edge_len, dim=1)
        return splits.select(split_dims)

    def split_random(self, branches: BranchStore) -> Split:
        """
        Randomly select an input dimensions to split.

        :param branches: The branches for which to determine the dimensions to split.
        :return: The splits to perform.
        """
        splits, is_invalid = self.propose_splits(branches)

        left_lbs, _ = splits.left_branch
        num_splits, batch_size, _ = left_lbs.shape

        branch_scores = torch.rand((num_splits, batch_size), generator=self._rng)
        branch_scores.masked_fill_(is_invalid, -torch.inf)
        split_dims = torch.argmax(branch_scores, dim=0)
        return splits.select(split_dims)

    def propose_splits(self, branches: BranchStore) -> tuple[Split, torch.Tensor]:
        """
        Creates all possible splits for a batch of branches.
        While continuous input attributes can be split arbitrarily often,
        categorical input attributes can only be split into the
        different values of the categorical attribute.
        Overall, the number of proposed splits is the number of dimensions
        of the :code:`combined_input_space`.

        :param branches: A batch of branches for which to determine splits.
        :return: A batch of splits and a tensor of booleans indicating invalid splits.
         Invalid splits are, for example, splits that split on the value of a categorical
         variable that was already split.

         The shape of the bounds inside the :code:`Split`
         is `(num_splits, batch_dim, input_dims)`.
         The index of the first dimension corresponds to the dimension of the
         combined input space that is split.
         This means that :code:`propose_splits(...)[i]` splits the i-th dimension
         of the combined input space.
         The tensor indicating invalid splits has the shape (num_splits, batch_dim).
         The i,j-th entry of the tensor is :code:`True` if the i-th split is invalid
         for the batch entry j.
        """
        lbs_flat = branches.in_lbs
        ubs_flat = branches.in_ubs

        def continuous_split(dim) -> tuple[Split, torch.Tensor]:
            # Split at midpoint, unless the dimension is unbounded.
            # If both sides are unbounded, split at 0.0.
            # If only one side is unbounded, use twice the finite bound as splitpoint.
            midpoint = (ubs_flat[:, dim] + lbs_flat[:, dim]) / 2
            midpoint = torch.nan_to_num(midpoint)  # this handles both sides unbounded

            lbs_bounded = lbs_flat[:, dim].isfinite()
            ubs_bounded = ubs_flat[:, dim].isfinite()
            splitpoint = torch.where(
                lbs_bounded ^ ubs_bounded,
                torch.where(
                    lbs_bounded,
                    torch.maximum(torch.ones(()), 2 * lbs_flat[:, dim]),
                    torch.minimum(-torch.ones(()), 2 * ubs_flat[:, dim]),
                ),
                midpoint,
            )

            # two splits: set lower bound to midpoint
            # and set upper bound to midpoint
            left_ubs = ubs_flat.detach().clone()
            right_lbs = lbs_flat.detach().clone()
            left_ubs[:, dim] = splitpoint
            right_lbs[:, dim] = splitpoint
            split = Split(
                left_branch=(lbs_flat, left_ubs), right_branch=(right_lbs, ubs_flat)
            )
            # The split is invalid if the bounds are already tight.
            is_invalid = torch.eq(ubs_flat[:, dim], lbs_flat[:, dim])
            return split, is_invalid

        def integer_split(dim) -> tuple[Split, torch.Tensor]:
            split, is_invalid = continuous_split(dim)
            left_lbs, left_ubs = split.left_branch
            right_lbs, right_ubs = split.right_branch
            # make sure to exclude the midpoint if it is already an integer
            left_ubs[:, dim] = torch.floor(left_ubs[:, dim])
            right_lbs[:, dim] = torch.floor(right_lbs[:, dim] + 1.0)
            split = Split(
                left_branch=(left_lbs, left_ubs), right_branch=(right_lbs, right_ubs)
            )
            return split, is_invalid

        def categorical_split(dims, val_i) -> tuple[Split, torch.Tensor]:
            """
            :param dims: The dimensions of the input space that together make up
             the categorical attribute.
            :param val_i: The index of the value to split on.
            """
            # two splits: set to the value and exclude the value

            # Left branch: set the value
            # => assign lbs and ubs of all other values to 0.0
            #    and assign lb and ub for val_i to 1.0.
            left_lbs = lbs_flat.detach().clone()
            left_ubs = ubs_flat.detach().clone()
            left_lbs[:, dims] = 0.0
            left_lbs[:, dims[val_i]] = 1.0
            left_ubs[:, dims] = 0.0
            left_ubs[:, dims[val_i]] = 1.0

            # Right branch: exclude the value
            # => set lb and ub for this value to 0.0.
            #    No change to other values.
            right_lbs = lbs_flat.detach().clone()
            right_ubs = ubs_flat.detach().clone()
            right_lbs[:, dims[val_i]] = 0.0
            right_ubs[:, dims[val_i]] = 0.0
            # But: may not exclude all values.
            # If there is only one value remaining after excluding this value,
            # we need to set the remaining value.
            dims_mask = torch.full_like(lbs_flat, fill_value=False, dtype=torch.bool)
            dims_mask[:, dims] = True
            unset_values = (right_lbs != right_ubs) & dims_mask
            num_unset_values = unset_values.float().sum(dim=1)
            num_unset_values.unsqueeze_(1)
            right_lbs = torch.where(
                num_unset_values == 1 & unset_values, 1.0, right_lbs
            )

            # If this value is already assigned (either set or excluded from assignment)
            # then we can't branch on this value any more.
            # Mark these splits as invalid.
            this_value_already_set = torch.eq(
                lbs_flat[:, dims[val_i]], ubs_flat[:, dims[val_i]]
            )

            return (
                Split((left_lbs, left_ubs), (right_lbs, right_ubs)),
                this_value_already_set,
            )

        splits: list[Split | None] = [None] * self._input_space.input_shape[0]
        is_invalid: list[torch.Tensor | None] = [None] * len(splits)
        for var, offset in zip(self._input_space.variables, self._input_space.offsets):
            var_domain = self._input_space.domain_of(var)
            if isinstance(var_domain, TensorInputSpace):
                for dim in range(self._input_space.dimensions_of(var)):
                    split, invalid = continuous_split(offset + dim)
                    splits[offset + dim] = split
                    is_invalid[offset + dim] = invalid
            elif isinstance(var_domain, TabularInputSpace):
                layout = var_domain.encoding_layout
                for attr in layout:
                    match var_domain.attribute_type(attr):
                        case TabularInputSpace.AttributeType.CONTINUOUS:
                            dim = offset + layout[attr]
                            split, invalid = continuous_split(dim)
                            splits[dim] = split
                            is_invalid[dim] = invalid
                        case TabularInputSpace.AttributeType.INTEGER:
                            dim = offset + layout[attr]
                            split, invalid = integer_split(dim)
                            splits[dim] = split
                            is_invalid[dim] = invalid
                        case TabularInputSpace.AttributeType.CATEGORICAL:
                            dims = [offset + dim for dim in layout[attr].values()]
                            for val_i, val in enumerate(
                                var_domain.attribute_values(attr)
                            ):
                                val_dim = offset + layout[attr][val]
                                split, invalid = categorical_split(dims, val_i)
                                splits[val_dim] = split
                                is_invalid[val_dim] = invalid
                        case _:
                            raise NotImplementedError()
            else:
                raise ValueError(
                    f"Unknown input space type {type(var_domain)}. "
                    f"Supported types: TensorInputSpace and TabularInputSpace."
                )

        return Split.stack(splits), torch.stack(is_invalid)

    def _get_branch_bounds(self, splits) -> tuple[
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        int,
        int,
    ]:
        left_lbs, left_ubs = splits.left_branch
        right_lbs, right_ubs = splits.right_branch
        num_splits, batch_size, input_size = left_lbs.shape
        left_lbs = left_lbs.reshape(-1, input_size)
        left_ubs = left_ubs.reshape_as(left_lbs)
        right_lbs = right_lbs.reshape_as(left_lbs)
        right_ubs = right_ubs.reshape_as(left_lbs)
        left_lbs = self._input_space.decompose(left_lbs)
        left_ubs = self._input_space.decompose(left_ubs)
        right_lbs = self._input_space.decompose(right_lbs)
        right_ubs = self._input_space.decompose(right_ubs)
        left_bounds = {var: (left_lbs[var], left_ubs[var]) for var in left_lbs}
        right_bounds = {var: (right_lbs[var], right_ubs[var]) for var in left_lbs}
        return left_bounds, right_bounds, num_splits, batch_size
