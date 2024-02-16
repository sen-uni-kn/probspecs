# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import itertools
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
import random
from math import prod
from typing import Sequence, Union

from frozendict import frozendict
import torch
import rust_enum

from .probability_distribution import ProbabilityDistribution
from ..input_space import CombinedInputSpace, TensorInputSpace
from ..utils.tensor_utils import to_tensor, TENSOR_LIKE


class BayesianNetwork(ProbabilityDistribution):
    """
    A Bayesian network.
    """

    # condition is a mapping from parent nodes to an event (lower + upper bounds)
    # distribution is the ProbabilityDistribution to use when condition matches
    # the values of all parents
    _ConditionalProbabilityTableEntry = namedtuple(
        "_ConditionalProbabilityTableEntry", ("condition", "distribution")
    )

    @dataclass(frozen=True, eq=False)
    class _Node:
        """
        A node of a :code:`BayesianNetwork`.
        """

        name: str
        parents: tuple["BayesianNetwork._Node", ...]
        conditional_probability_table: tuple[
            "BayesianNetwork._ConditionalProbabilityTableEntry", ...
        ]

        def __eq__(self, other):
            return self is other

        def __hash__(self):
            return super().__hash__()

    def __init__(self, nodes: Sequence["BayesianNetwork._Node"]):
        """Creates a new :code:`BayesianNetwork`"""
        nodes = tuple(nodes)
        if len(nodes) == 0:
            raise ValueError("A BayesianNetwork needs to have at least one node.")

        node_names = set()
        for node in nodes:
            if node.name in node_names:
                raise ValueError(f"Node name {node.name} is not unique.")
            node_names.add(node.name)

        # order the nodes from sources to sinks
        processed_nodes = set()
        ordered_nodes = []
        while len(processed_nodes) < len(nodes):
            for node in nodes:
                if node in processed_nodes:
                    continue
                if processed_nodes.issuperset(node.parents):
                    processed_nodes.add(node)
                    ordered_nodes.append(node)
        self.__nodes = tuple(ordered_nodes)
        self.__original_order = nodes

        self.__node_event_shapes = {
            node: node.conditional_probability_table[0].distribution.event_shape
            for node in nodes
        }

        # Check that the bounds in all conditional probability tables match
        # the parent node event shapes
        for node in nodes:
            for condition, _ in node.conditional_probability_table:
                for parent, (lower, upper) in condition.items():
                    try:
                        lower.reshape(self.__node_event_shapes[parent])
                        upper.reshape(self.__node_event_shapes[parent])
                    except RuntimeError as e:
                        raise ValueError(
                            f"Bounds for parent node {parent.name} in conditional "
                            f"probability table of node {node.name} do not match the "
                            f"event shape of {parent.name}."
                        ) from e

        self.__output_space = CombinedInputSpace(
            {
                node.name: TensorInputSpace(
                    torch.full(self.__node_event_shapes[node], fill_value=-torch.inf),
                    torch.full(self.__node_event_shapes[node], fill_value=torch.inf),
                )
                for node in self.__original_order
            }
        )

    def probability(self, event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Example computation for the following graph:
        # A     B
        #  ↘   ↙
        #    C
        #  ↙  ↘
        # D    E
        # We want to compute the probability of the event
        # [a1, a2] x [b1, b2] x [c1, c2] x [d1, d2] x [e1, e2].
        # This corresponds to
        # P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2] ∧ D ∈ [d1, d2] ∧ E ∈ [e1, e2])
        #   | Chain rule of probability
        # = P(E ∈ [e1, e2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2] ∧ D ∈ [d1, d2])
        #   * P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2] ∧ D ∈ [d1, d2])
        #   | Conditional independence
        # = P(E ∈ [e1, e2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   * P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2] ∧ D ∈ [d1, d2])
        #   | Chain rule of probability
        # = P(E ∈ [e1, e2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   * P(D ∈ [d1, d2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   * P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   | Chain rule of probability
        # = P(E ∈ [e1, e2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   * P(D ∈ [d1, d2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   * P(C ∈ [c1, c2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2])
        #   * P(A ∈ [a1, a2] ∧ B ∈ [b1, b2])
        #   | Conditional independence
        # = P(E ∈ [e1, e2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   * P(D ∈ [d1, d2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   * P(C ∈ [c1, c2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2])
        #   * P(A ∈ [a1, a2])
        #   * P(B ∈ [b1, b2])
        #
        # We can compute each conditional term using the conditional probability table
        # and the law of total probability.
        # For example,
        # P(E ∈ [e1, e2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   | Law of total probability
        #   | [ai1, ai2] x [bi1, bi2] x [ci1, ci2] is the i-th condition
        #   | of E's conditional probability table
        # = Σ_i P(E ∈ [e1, e2] | A ∈ [a1, a2] ∩ [ai1, ai2] ∧ B ∈ [b1, b2] ∩ [bi1, bi2]
        #                        ∧ C ∈ [c1, c2] ∩ [ci1, ci2])
        #       * P(A ∈ [ai1, ai2] ∧ B ∈ [bi1, bi2] ∧ C ∈ [ci1, ci2]
        #           | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   | Definition of conditional probability
        # = Σ_i P(E ∈ [e1, e2] | A ∈ [a1, a2] ∩ [ai1, ai2] ∧ B ∈ [b1, b2] ∩ [bi1, bi2]
        #                        ∧ C ∈ [c1, c2] ∩ [ci1, ci2])
        #       * P(A ∈ [ai1, ai2] ∧ B ∈ [bi1, bi2] ∧ C ∈ [ci1, ci2]
        #           ∧ A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #       / P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        # = Σ_i P(E ∈ [e1, e2] | A ∈ [a1, a2] ∩ [ai1, ai2] ∧ B ∈ [b1, b2] ∩ [bi1, bi2]
        #                        ∧ C ∈ [c1, c2] ∩ [ci1, ci2])
        #       * P(A ∈ [a1, a2] ∩ [ai1, ai2] ∧ B ∈ [b1, b2] ∩ [bi1, bi2]
        #           ∧ C ∈ [c1, c2] ∩ [ci1, ci2])
        #       / P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        # Note that because of the second factor of the multiplication, a term in the
        # sum is zero if [a1, a2] ∩ [ai1, ai2], [b1, b2] ∩ [bi1, bi2], or
        # [c1, c2] ∩ [ci1, ci2] is empty (condition doesn't match),
        # even though the conditional probability in the first factor is undefined
        # in this case.

        @dataclass(frozen=True, eq=False)
        class AtomicEvent:
            """An event of one node."""

            node: "BayesianNetwork._Node"
            lower: torch.Tensor
            upper: torch.Tensor

            def as_hashable(
                self,
            ) -> tuple["BayesianNetwork._Node", tuple[float, ...], tuple[float, ...]]:
                return (
                    self.node,
                    tuple(self.lower.flatten().tolist()),
                    tuple(self.upper.flatten().tolist()),
                )

        # Term like P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2]) appear several times
        # in the overall probability.
        # To avoid computing them multiple times, we cache them.
        # We also cache values like P(A ∈ [a1, a2] ∩ [ai1, ai2]
        # ∧ B ∈ [b1, b2] ∩ [bi1, bi2] ∧ C ∈ [c1, c2] ∩ [ci1, ci2]) in case conditions
        # appear in multiple conditional probability tables.
        #
        # We represent conjuncts as tuples that are always ordered according to
        # self.__nodes
        conjuncts_cache: dict[
            tuple[
                tuple[BayesianNetwork._Node, tuple[float, ...], tuple[float, ...]], ...
            ],
            torch.Tensor,
        ] = {}

        def probability_of_conjunction(
            event_: tuple[AtomicEvent, ...],
        ) -> torch.Tensor | float:
            if len(event_) == 0:
                return 1.0

            event_hashable = tuple(ae.as_hashable() for ae in event_)
            if event_hashable not in conjuncts_cache:
                # apply the chain rule for the last node in event_
                # (event_ is always ordered from sources to sinks)
                conditional = probability_conditioned_on_parents(event_)
                parents = probability_of_conjunction(event_[:-1])
                conjuncts_cache[event_hashable] = conditional * parents
            return conjuncts_cache[event_hashable]

        def probability_conditioned_on_parents(
            event_: tuple[AtomicEvent, ...],
        ) -> torch.Tensor | float:
            """Compute P(event_[-1] | event[:-1])."""
            if len(event_) == 0:
                return 1.0

            subject = event_[-1]
            others = event_[:-1]
            node = subject.node
            # Eliminate conditionally independent nodes from event_.
            # Since we walk from sinks to sources (see below),
            # no descendants of node can be contained in event_.
            predecessors_stack = list(node.parents)
            predecessors = set()
            while len(predecessors_stack) > 0:
                predecessor = predecessors_stack.pop()
                predecessors_stack.extend(predecessor.parents)
                predecessors.add(predecessor)
            parent_event = tuple(ae for ae in others if ae.node in predecessors)

            node_prob = torch.zeros(1)
            for condition, distribution in node.conditional_probability_table:
                # Check if this table entry applies.
                # It applies if the table entry condition intersects with
                # the parent event.
                # Here we instead check if condition and parent_event are disjoint
                # (do not intersect).
                # This is the case if the bounds of condition and parent_event
                # are disjoint for any variable in any dimension.
                if any(
                    torch.any(
                        (ae.lower > condition[ae.node][1])  # lower > upper
                        | (ae.upper < condition[ae.node][0])  # upper < lower
                    )
                    for ae in parent_event
                    if ae.node in condition
                ):
                    # This term of the sum is zero
                    continue

                # First term: P(subject | condition)
                # Actually: P(subject | condition ∧ parent_event),
                # but since we established that condition and parent_event intersect,
                # these two probabilities are the same, since we interpret condition
                # as a discrete indicator variable.
                cond_prob = distribution.probability((subject.lower, subject.upper))
                # intersection of condition and parent_event
                intersection = tuple(
                    AtomicEvent(
                        p_ae.node,
                        torch.maximum(p_ae.lower, condition[p_ae.node][0]),
                        torch.minimum(p_ae.upper, condition[p_ae.node][1]),
                    )
                    if p_ae.node in condition
                    else p_ae
                    for p_ae in parent_event
                )
                intersection_prob = probability_of_conjunction(intersection)
                # all terms are divided by P(parent_event), so we factor that out
                node_prob += cond_prob * intersection_prob

            parents_prob = probability_of_conjunction(parent_event)
            return node_prob / parents_prob

        lower, upper = event
        lower = self.__output_space.decompose(torch.atleast_2d(lower))
        upper = self.__output_space.decompose(torch.atleast_2d(upper))
        event = tuple(
            AtomicEvent(
                node,
                lower[node.name].reshape(self.__node_event_shapes[node]),
                upper[node.name].reshape(self.__node_event_shapes[node]),
            )
            for node in self.__nodes
        )

        # walk from sinks to sources
        prob = probability_conditioned_on_parents(event)
        event = event[:-1]
        while len(event) > 0:
            p = probability_conditioned_on_parents(event)
            # if p is zero for some node, previous conditional probabilities
            # are nan, due to division by zero.
            # However, we want to treat the overall product as 0.0 instead of nan.
            prob = torch.where(p != 0, prob * p, 0.0)
            event = event[:-1]
        return prob

    def sample(self, num_samples: int, seed=None) -> torch.Tensor:
        cache = {}
        seed_rng = random.Random()
        seed_rng.seed(seed)
        for node in self.__nodes:
            # self.__nodes is ordered from sources to sinks.
            # Therefore, the parents of a node already have values in the cache
            # when we process the node.
            # To support batch processing, we generate samples from all possible
            # distributions and select one depending on the batch element.
            sample = torch.full(
                (num_samples, prod(self.__node_event_shapes[node])),
                fill_value=torch.nan,
            )
            for condition, distribution in node.conditional_probability_table:
                seed = seed_rng.randint(0, 2**32 - 1)
                sample_j = distribution.sample(num_samples, seed)
                sample_j = sample_j.reshape((num_samples, -1))
                select = torch.full((num_samples,), fill_value=True)
                for parent in node.parents:
                    lower, upper = condition[parent]
                    select_k = (lower <= cache[parent]) & (cache[parent] <= upper)
                    select_k = torch.all(select_k.flatten(1), dim=1)
                    select &= select_k
                select.unsqueeze_(-1)
                sample = torch.where(select, sample_j, sample)
            cache[node] = sample
        return torch.hstack([cache[node] for node in self.__original_order])

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size(self.__output_space.input_shape)

    @property
    def output_space(self) -> CombinedInputSpace:
        """
        The space of the values sampled from this :code:`BayesianNetwork`.

        The output space contains information, such as the position where
        values of nodes are encoded in the flattened output space.

        :code:`BayesianNetwork` does not store bounds on it's nodes.
        Therefore, the bounds of the :code:`output_space` are
        :math:`[-\\infty, \\infty]` for all dimensions.
        """
        return self.__output_space

    class Factory:
        """
        Create Bayesian networks.

        Node order
        ==========
        The order of nodes determines the order in which values of nodes appear
        in the output space of the BayesianNetwork.
        By default, the order of nodes is determined by their creation order
        using the :code:`new_node` method.
        You can reorder nodes using the :code:`reorder_nodes` method.
        """

        @rust_enum.enum
        class EventSpace:
            """
            A description of the event space of a node.
            """

            Unbounded = rust_enum.Case()
            Continuous = rust_enum.Case(lower=torch.Tensor, upper=torch.Tensor)
            Discrete = rust_enum.Case(values=tuple[torch.Tensor, ...])

        class Node:
            def __init__(self, name: str):
                self.__name = name
                self.__parents = []
                self.__event_space: "BayesianNetwork.Factory.EventSpace" = (
                    BayesianNetwork.Factory.EventSpace.Unbounded()
                )
                self.__conditional_probability_table: tuple[
                    tuple[
                        dict[str, tuple[torch.Tensor, torch.Tensor]],
                        ProbabilityDistribution,
                    ],
                    ...,
                ] | None = None

            @property
            def name(self) -> str:
                return self.__name

            def add_parent(self, node: Union[str, "BayesianNetwork.Factory.Node"]):
                self._check_cond_prob_table_not_created("node parents")

                if not isinstance(node, str):
                    node = node.name
                self.__parents.append(node)

            def remove_parent(self, node: Union[str, "BayesianNetwork.Factory.Node"]):
                self._check_cond_prob_table_not_created("node parents")

                if not isinstance(node, str):
                    node = node.name
                self.__parents.remove(node)

            def set_parents(self, *nodes: Union[str, "BayesianNetwork.Factory.Node"]):
                self._check_cond_prob_table_not_created("node parents")

                self.__parents = []
                for node in nodes:
                    self.add_parent(node)

            @property
            def parents(self) -> tuple[str, ...]:
                return tuple(self.__parents)

            def unbounded_event_space(self):
                """
                Sets this node to have an unbounded event space.
                """
                self._check_cond_prob_table_not_created("event space")
                self.__event_space = BayesianNetwork.Factory.EventSpace.Unbounded()

            def continuous_event_space(self, lower: TENSOR_LIKE, upper: TENSOR_LIKE):
                """
                Sets this node to have a continuous event space in the form
                of a hyper-rectangle with minimal elements :code:`lower`
                and maximal elements :code:`upper`.
                """
                self._check_cond_prob_table_not_created("event space")

                lower = to_tensor(lower)
                upper = to_tensor(upper)
                self.__event_space = BayesianNetwork.Factory.EventSpace.Continuous(
                    lower, upper
                )

            def discrete_event_space(self, *values: TENSOR_LIKE):
                """
                Sets this node to have a discrete event space with the
                given values as options.
                """
                self._check_cond_prob_table_not_created("event space")

                values = tuple(to_tensor(val) for val in values)
                self.__event_space = BayesianNetwork.Factory.EventSpace.Discrete(values)

            @property
            def event_space(self) -> "BayesianNetwork.Factory.EventSpace":
                """
                The event space of this node.
                By default, the event space is unbounded.
                You can change the event space using the :code:`continuous_event_space`
                :code:`discrete_event_space` and :code:`unbounded_event_space` methods.
                """
                return self.__event_space

            def set_conditional_probability(
                self,
                condition: dict[
                    Union[str, "BayesianNetwork.Factory.Node"],
                    tuple[TENSOR_LIKE, TENSOR_LIKE] | TENSOR_LIKE,
                ],
                distribution: ProbabilityDistribution,
            ):
                """
                Set the conditional probability distribution for the case
                when the values of the parent nodes are in the events
                in :code:`condition`.

                Once a conditional probability was set, the parents of a node
                can not be changed.

                :param condition: A mapping from parent node (names) to events.
                 Events can either be single values for discrete event spaces,
                 or tuples of minimal (lower) and maximal (upper) elements.
                 In this case, the minimal and maximal elements describe a rectangular
                 event (a set) in the event space of the parent node.
                 If the values produced by the parent nodes all lie in their respective
                 condition events, the value of this node is determined
                 by :code:`distribution`
                 The conditions need to be disjoint from all previously set conditions
                 in a conditional probability.
                :param distribution: The probability distribution that determines the
                 value of this node when :code:`condition` matches.
                """
                for node in condition:
                    if not isinstance(node, str):
                        node = node.name
                    if node not in self.__parents:
                        raise ValueError(f"Unknown parent node in condition: {node}")

                condition = {
                    node if isinstance(node, str) else node.name: event
                    for node, event in condition.items()
                }

                for node, event in condition.items():
                    if isinstance(event, tuple):
                        lower, upper = event
                        lower = to_tensor(lower)
                        upper = to_tensor(upper)
                    else:
                        lower = upper = to_tensor(event)
                    condition[node] = (lower, upper)

                # check disjointness with other table entries
                if self.__conditional_probability_table is not None:
                    for other_condition, _ in self.__conditional_probability_table:
                        any_disjoint = False
                        for parent, event in condition.items():
                            if parent in other_condition:
                                lower, upper = event
                                other_lower, other_upper = other_condition[parent]
                                if torch.any(
                                    (lower >= other_upper) | (upper <= other_lower)
                                ):
                                    any_disjoint = True
                                    break
                        if not any_disjoint:
                            raise ValueError(
                                f"Condition {condition} is not disjoint from previously "
                                f"registered condition {other_condition}."
                            )

                        all_equal = True
                        for parent, event in condition.items():
                            if parent in other_condition:
                                lower, upper = event
                                other_lower, other_upper = other_condition[parent]
                                if torch.any(
                                    (lower != other_upper) | (upper != other_lower)
                                ):
                                    all_equal = False
                                    break
                        if all_equal:
                            raise ValueError(
                                f"Condition {condition} is identical to previously "
                                f"registered condition {other_condition}."
                            )

                if self.__conditional_probability_table is not None:
                    for _, other_distribution in self.__conditional_probability_table:
                        if other_distribution.event_shape != distribution.event_shape:
                            raise ValueError(
                                f"Distribution {distribution} has a different event shape "
                                f"than previously supplied distribution "
                                f"{other_distribution}."
                            )

                if self.__conditional_probability_table is None:
                    self.__conditional_probability_table = []
                self.__conditional_probability_table.append((condition, distribution))

            @property
            def conditional_probability_table(
                self,
            ) -> tuple[
                tuple[
                    dict[str, tuple[torch.Tensor, torch.Tensor]],
                    ProbabilityDistribution,
                ],
                ...,
            ]:
                if self.__conditional_probability_table is None:
                    raise RuntimeError("Conditional probability table not set.")
                return tuple(self.__conditional_probability_table)

            def _check_cond_prob_table_not_created(self, attribute_name: str):
                if self.__conditional_probability_table is not None:
                    raise ValueError(
                        f"The {attribute_name} can not be changed after the conditional "
                        "probability table was created."
                    )

        def __init__(self):
            self.__nodes: OrderedDict[
                str, "BayesianNetwork.Factory.Node"
            ] = OrderedDict()

        def new_node(self, name: str) -> "BayesianNetwork.Factory.Node":
            if name in self.__nodes:
                raise ValueError(
                    f"Node name {name} already used. Nodes need to have unique names."
                )
            node = BayesianNetwork.Factory.Node(name)
            self.__nodes[name] = node
            return node

        def new_nodes(self, *names: str) -> tuple["BayesianNetwork.Factory.Node", ...]:
            return tuple(self.new_node(name) for name in names)

        @property
        def nodes(self) -> tuple[str, ...]:
            return tuple(self.__nodes)

        def reorder_nodes(self, new_order: tuple[str, ...]):
            for name in new_order:
                self.__nodes.move_to_end(name)

        def __getitem__(self, node_name: str) -> "BayesianNetwork.Factory.Node":
            """
            Retrieves the node named :code:`node_name`.
            """
            return self.__nodes[node_name]

        def create(self) -> "BayesianNetwork":
            """
            Creates a Bayesian network with the previously
            added and configured nodes.

            This method does not change the state of the factory or the nodes.
            Therefore, subsequent calls of :code:`create` without intermediate
            changes to the factory object will create equivalent :code:`BayesianNetworks`.

            :return: A new :code:`BayesianNetwork`.
            """
            # Check that conditional probability tables cover the entire
            # parent event space.
            for node in self.__nodes.values():
                if len(node.parents) == 0:
                    continue
                # maintain a partition of the parents space to determine if the
                # entirety of the parents space is covered
                partition = [{p: self.__nodes[p].event_space for p in node.parents}]

                def contains(
                    event: tuple[torch.Tensor, torch.Tensor],
                    other: "BayesianNetwork.Factory.EventSpace",
                ) -> bool:
                    lower, upper = event
                    match other:
                        case BayesianNetwork.Factory.EventSpace.Continuous(l, u):
                            return torch.all((lower <= l) & (u <= upper))
                        case BayesianNetwork.Factory.EventSpace.Discrete(vals):
                            return all(
                                torch.all((lower <= v) & (v <= upper)) for v in vals
                            )
                        case BayesianNetwork.Factory.EventSpace.Unbounded():
                            return torch.all(lower.isneginf() & upper.isposinf())
                        case _:
                            raise NotImplementedError()

                for condition, _ in node.conditional_probability_table:
                    new_partition = []
                    for part in partition:
                        # split part into covered/not covered, then filter out
                        # what is entirely covered by condition.
                        splits = {}
                        for p, event_space in part.items():
                            lower, upper = condition[p]
                            match event_space:
                                case BayesianNetwork.Factory.EventSpace.Unbounded():
                                    l_ = torch.full(lower.shape, fill_value=-torch.inf)
                                    u_ = torch.full(lower.shape, fill_value=torch.inf)
                                    event_space = (
                                        BayesianNetwork.Factory.EventSpace.Continuous(
                                            l_, u_
                                        )
                                    )
                            match event_space:
                                case BayesianNetwork.Factory.EventSpace.Continuous(
                                    l, u
                                ):
                                    lower_ = lower.flatten()
                                    upper_ = upper.flatten()
                                    l_ = l.flatten()
                                    u_ = u.flatten()
                                    segments = []
                                    for i in range(lower_.size(0)):
                                        segments_i = []
                                        if u_[i] <= lower_[i] or upper_[i] <= l_[i]:
                                            # no intersection
                                            segments_i.append((l_[i], u_[i]))
                                        else:
                                            if l_[i] < lower_[i]:
                                                segments_i.append((l_[i], lower_[i]))
                                            if upper_[i] < u_[i]:
                                                segments_i.append((upper_[i], u_[i]))
                                            segments_i.append(
                                                (
                                                    max(l_[i], lower_[i]),
                                                    min(u_[i], upper_[i]),
                                                )
                                            )
                                        segments.append(segments_i)
                                    splits[p] = [
                                        BayesianNetwork.Factory.EventSpace.Continuous(
                                            torch.tensor([l_ for l_, _ in segments_i]),
                                            torch.tensor([u_ for _, u_ in segments_i]),
                                        )
                                        for segments_i in itertools.product(*segments)
                                    ]
                                case BayesianNetwork.Factory.EventSpace.Discrete(vals):
                                    contained = []
                                    not_contained = []
                                    for v in vals:
                                        if torch.all((lower <= v) & (v <= upper)):
                                            contained.append(v)
                                        else:
                                            not_contained.append(v)
                                    splits[p] = []
                                    if len(contained) > 0:
                                        splits[p].append(
                                            BayesianNetwork.Factory.EventSpace.Discrete(
                                                contained
                                            )
                                        )
                                    if len(not_contained) > 0:
                                        splits[p].append(
                                            BayesianNetwork.Factory.EventSpace.Discrete(
                                                not_contained
                                            )
                                        )
                        splits = [
                            [(p, s) for s in segments] for p, segments in splits.items()
                        ]
                        for segments in itertools.product(*splits):
                            new_part = {p: s for p, s in segments}
                            if all(
                                contains(condition[p], new_part[p])
                                for p in node.parents
                            ):
                                # the new part is covered
                                continue
                            else:
                                new_partition.append(new_part)
                    partition = new_partition
                if len(partition) > 0:
                    raise ValueError(
                        f"Conditional probability table does not cover the entire "
                        f"combined event space of the parents of {node.name}. "
                        f"Not covered: {partition}."
                    )

            nodes = {}
            processed: set[str] = set()
            while len(nodes) < len(self.__nodes):
                for node in self.__nodes.values():
                    if node.name in processed:
                        continue
                    elif processed.issuperset(node.parents):
                        # replace node names by nodes
                        parents = tuple(nodes[p] for p in node.parents)

                        # also replace node names by nodes in the conditional probability table
                        cond_prob_table = tuple(
                            BayesianNetwork._ConditionalProbabilityTableEntry(
                                {nodes[p]: event for p, event in condition.items()},
                                distribution,
                            )
                            for condition, distribution in node.conditional_probability_table
                        )
                        new_node = BayesianNetwork._Node(
                            node.name,
                            parents,
                            cond_prob_table,
                        )
                        nodes[node.name] = new_node
                        processed.add(node.name)
            # recreate order from self.__nodes
            nodes = tuple(nodes[node_name] for node_name in self.__nodes)
            return BayesianNetwork(nodes)
