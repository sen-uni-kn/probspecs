# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import itertools
import operator
from collections import OrderedDict
from dataclasses import dataclass
import random
from functools import reduce
from math import prod
from typing import Sequence, Union, NamedTuple

import torch
import rust_enum

from .probability_distribution import ProbabilityDistribution
from ..input_space import CombinedInputSpace, TensorInputSpace
from ..utils.tensor_utils import to_tensor, TENSOR_LIKE


__all__ = ["BayesianNetwork"]


class _Event:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        self.lb = torch.atleast_1d(lb)
        self.ub = torch.atleast_1d(ub)

    def __len__(self):
        return 2

    def __getitem__(self, item):
        if item == 0:
            return self.lb
        elif item == 1:
            return self.ub
        else:
            raise IndexError

    def __iter__(self):
        return iter((self.lb, self.ub))

    def __eq__(self, other):
        if self.lb.shape != other.lb.shape:
            return False
        return torch.all((self.lb == other.lb) & (self.ub == other.ub))

    def __hash__(self):
        return hash(
            (
                tuple(self.lb.detach().tolist()),
                tuple(self.ub.detach().tolist()),
            )
        )


class _ConditionalProbabilityTableEntry(NamedTuple):
    """
    An entry of a conditional probability table.

    - condition is a mapping from parent nodes to an event (lower + upper bounds)
    - distribution is the ProbabilityDistribution to use when condition matches
      the values of all parents
    """

    condition: dict["_Node", _Event]
    distribution: ProbabilityDistribution


@dataclass(frozen=True, eq=False)
class _Node:
    """
    A node of a :code:`BayesianNetwork`.

    Nodes can be hidden (also called latent).
    If hidden, nodes don't appear in samples unless explicitly requested
    using :code:`BayesianNetwork.include_hidden`.
    """

    name: str
    parents: tuple["_Node", ...]
    conditional_probability_table: tuple[_ConditionalProbabilityTableEntry, ...]
    hidden: bool

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        parent_names = ", ".join([parent.name for parent in self.parents])
        return f"_Node(name={self.name}, parents={{{parent_names}}}, hidden={self.hidden}, id={id(self)})"


class BayesianNetwork(ProbabilityDistribution):
    """
    A Bayesian network.
    """

    def __init__(self, nodes: Sequence[_Node], dtype: torch.dtype = torch.double):
        """Creates a new :code:`BayesianNetwork`"""
        if len(nodes) == 0:
            raise ValueError("A BayesianNetwork needs to have at least one node.")

        node_names = set()
        for node in nodes:
            if node.name in node_names:
                raise ValueError(f"Node name {node.name} is not unique.")
            node_names.add(node.name)

        # order the nodes from sources to sinks
        nodes_ordered = self._order_nodes(nodes)
        nodes, nodes_ordered = self._unify_conditions(nodes, nodes_ordered, dtype)
        self.__nodes = nodes_ordered
        self.__original_order = nodes

        # Compute the predecessors of all nodes.
        # The predecessors are used for computing probabilities.
        self.__predecessors = self._build_predecessors(self.__nodes)

        self.__node_event_shapes = {
            node: node.conditional_probability_table[0].distribution.event_shape
            for node in self.__nodes
        }

        # Check that the bounds in all conditional probability tables match
        # the parent node event shapes
        for node in self.__nodes:
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
        self.__include_hidden = False

        # For matching event elements to nodes
        self.__output_space = CombinedInputSpace(
            {
                node.name: TensorInputSpace(
                    torch.full(self.__node_event_shapes[node], fill_value=-torch.inf),
                    torch.full(self.__node_event_shapes[node], fill_value=torch.inf),
                )
                for node in self.__original_order
                if not node.hidden
            }
        )
        self.__output_space_including_hidden = CombinedInputSpace(
            {
                node.name: TensorInputSpace(
                    torch.full(self.__node_event_shapes[node], fill_value=-torch.inf),
                    torch.full(self.__node_event_shapes[node], fill_value=torch.inf),
                )
                for node in self.__original_order
            }
        )
        self.__dtype = dtype

    @staticmethod
    def _unify_conditions(
        nodes: Sequence[_Node], nodes_ordered: Sequence[_Node], dtype: torch.dtype
    ) -> tuple[tuple[_Node, ...], tuple[_Node, ...]]:
        """
        Replace equal events in conditional probability tables by
        identical _Event objects.
        This improves caching speed in :code:`probability`.

        Also converts all conditions to :code:`self.__dtype`.

        :param nodes: The nodes to process
        :param nodes_ordered: The same nodes, but ordered from sources to sinks
        :return: The new nodes, ordered as `nodes` and the new nodes, ordered
         as `nodes_ordered`.
        """
        events = {}  # stores the canonical event instance
        converted: dict[_Node, _Node] = {}  # old nodes to new nodes

        def lookup_event(event: _Event) -> _Event:
            if event not in events:
                lb, ub = event
                lb, ub = lb.to(dtype), ub.to(dtype)
                events[event] = _Event(lb, ub)
            return events[event]

        def convert_table_entry(
            entry: _ConditionalProbabilityTableEntry,
        ) -> _ConditionalProbabilityTableEntry:
            condition, distribution = entry
            condition = {
                converted[parent]: lookup_event(event)
                for parent, event in condition.items()
            }
            return _ConditionalProbabilityTableEntry(condition, distribution)

        # walk from sources to sinks so that parent nodes were already processed
        for node in nodes_ordered:
            table = tuple(map(convert_table_entry, node.conditional_probability_table))
            parents = tuple(converted[node] for node in node.parents)
            new_node = _Node(node.name, parents, table, node.hidden)
            converted[node] = new_node

        new_nodes_ordered = tuple(converted.values())  # dict remembers insertion order
        new_nodes = tuple(converted[node] for node in nodes)
        return new_nodes, new_nodes_ordered

    @staticmethod
    def _order_nodes(nodes: Sequence[_Node]) -> tuple[_Node, ...]:
        """Orders nodes from sources to sinks"""
        processed_nodes = set()
        ordered_nodes = []
        while len(processed_nodes) < len(nodes):
            for node in nodes:
                if node in processed_nodes:
                    continue
                if processed_nodes.issuperset(node.parents):
                    processed_nodes.add(node)
                    ordered_nodes.append(node)
        return tuple(ordered_nodes)

    @staticmethod
    def _build_predecessors(
        ordered_nodes: tuple[_Node, ...],
    ) -> dict[_Node, set[_Node]]:
        """
        Compute the predecessors of all nodes.
        :param ordered_nodes: Nodes, ordered from sources to sinks.
        """
        predecessors = {}
        for node in ordered_nodes:
            node_predecessors = set()
            for parent in node.parents:
                node_predecessors.add(parent)
                node_predecessors.update(predecessors[parent])
            predecessors[node] = node_predecessors
        return predecessors

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
        #   | condition_i is the i-th condition of E's conditional probability table
        # = Σ_i P(E ∈ [e1, e2] | condition_i ∧ A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #       * P(condition_i | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #   | Definition of conditional probability
        # = Σ_i P(E ∈ [e1, e2] | condition_i ∧ A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #       * P(condition_i ∧ A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #       / P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        # A problem with the above expression is that we don't know
        # P(E ∈ [e1, e2] | condition_i ∧ A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2]).
        # However, we know P(E ∈ [e1, e2] | condition_i) since it an entry in
        # E's conditional probability table.
        # We can think of selecting a row in E's conditional probability table as a
        # separate deterministic node in the Bayesian network.
        # This node is then the only parent node of E, with E's parents being the parents
        # of the new deterministic node.
        # With this,
        # P(E ∈ [e1, e2] | condition_i ∧ A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        # = P(E ∈ [e1, e2] | condition_i)
        # due to conditional independence of E given E's (virtual) single parent node.
        # Therefore,
        # P(E ∈ [e1, e2] | A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        # = Σ_i P(E ∈ [e1, e2] | condition_i)
        #       * P(condition_i ∧ A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #       / P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2])
        #
        # Note on hidden nodes:
        # Work just as other nodes, but when :code:`self.include_hidden` is false, we insert
        # the hidden node's entire event space into event.

        @dataclass(frozen=True)
        class AtomicEvent:
            """An event of one node."""

            node: _Node
            lower: torch.Tensor
            upper: torch.Tensor
            # Identifiers for the conditions that were conjoined with the root event
            # (the argument `event` to `probability`) to obtain this AtomicEvent.
            # Used for hashing, as (together with node) it uniquely identifies a
            # conjunction during the course of the algorithm.
            applied_conditions: frozenset[int] = frozenset()

            def __eq__(self, other):
                return (
                    self.node == other.node
                    and self.applied_conditions == other.applied_conditions
                )

            def __hash__(self):
                return hash((self.node, self.applied_conditions))

            def intersect(self, condition: dict[_Node, _Event]) -> "AtomicEvent":
                if self.node not in condition:
                    return self
                node_condition = condition[self.node]
                now_applied = frozenset([id(node_condition), *self.applied_conditions])
                condition_lower, condition_upper = node_condition
                if self.lower is not None:
                    new_lower = torch.maximum(self.lower, condition_lower)
                else:
                    new_lower = condition_lower
                if self.upper is not None:
                    new_upper = torch.minimum(self.upper, condition_upper)
                else:
                    new_upper = condition_upper
                # the above may make new_lower > new_upper, but we can't fix this
                # here, because we would have to differentiate discrete and continuous
                # event spaces.
                # ProbabilityDistributions have to handle these cases properly.
                return AtomicEvent(
                    self.node, new_lower, new_upper, applied_conditions=now_applied
                )

        lower, upper = event
        lower, upper = lower.to(self.dtype), upper.to(self.dtype)
        out_space = (
            self.__output_space
            if not self.include_hidden
            else self.__output_space_including_hidden
        )
        lower = out_space.decompose(torch.atleast_2d(lower))
        upper = out_space.decompose(torch.atleast_2d(upper))
        event = tuple(
            (
                AtomicEvent(
                    node,
                    lower[node.name].reshape(-1, *self.__node_event_shapes[node]),
                    upper[node.name].reshape(-1, *self.__node_event_shapes[node]),
                )
                if not node.hidden or self.include_hidden
                else AtomicEvent(
                    node,
                    torch.full(
                        (1, *self.__node_event_shapes[node]), fill_value=-torch.inf
                    ),
                    torch.full(
                        (1, *self.__node_event_shapes[node]), fill_value=torch.inf
                    ),
                )
            )
            for node in self.__nodes
        )

        # Term like P(A ∈ [a1, a2] ∧ B ∈ [b1, b2] ∧ C ∈ [c1, c2]) appear several times
        # in the overall probability.
        # To avoid computing them multiple times, we cache them.
        # We represent conjuncts as tuples of atomic events.
        conjuncts_cache: dict[
            tuple[AtomicEvent, ...],
            torch.Tensor,
        ] = {}

        def probability_of_conjunction(
            event_: tuple[AtomicEvent, ...],
        ) -> torch.Tensor:
            if len(event_) == 0:
                return torch.ones((), dtype=self.dtype)

            def compute_intersection_prob():
                # Apply the chain rule for the last node in event_
                # (event_ is always ordered from sources to sinks)
                conditional = probability_conditioned_on_parents(event_)
                parents = probability_of_conjunction(event_[:-1])
                # If we apply the chain rule to compute the probability of an
                # empty intersection, the probability condition on parents can be
                # undefined, if the intersection of the parent events is zero.
                # Practically, it is nan in this case.
                # However, the probability of the intersection is not undefined but zero.
                # We work around this case by setting the probability of the intersection
                # to zero if the probability of the parent's intersection is zero.
                zero = torch.zeros((), dtype=self.dtype)
                return torch.where(parents.isclose(zero), zero, conditional * parents)

            if event_ not in conjuncts_cache:
                conjuncts_cache[event_] = compute_intersection_prob()
            return conjuncts_cache[event_]

        def probability_conditioned_on_parents(
            event_: tuple[AtomicEvent, ...],
        ) -> torch.Tensor:
            """Compute P(event_[-1] | event[:-1])."""
            if len(event_) == 0:
                return torch.ones((), dtype=self.dtype)

            subject = event_[-1]
            others = event_[:-1]
            node = subject.node

            predecessors = self.__predecessors[node]
            parent_event = tuple(ae for ae in others if ae.node in predecessors)

            parents_prob = probability_of_conjunction(parent_event)
            node_prob = torch.zeros(subject.lower.size(0), dtype=self.dtype)
            for condition, distribution in node.conditional_probability_table:
                # First term: P(subject | condition)
                subject_given_condition = distribution.probability(
                    (subject.lower, subject.upper)
                )
                subject_given_condition = subject_given_condition.to(self.dtype)
                # Second term: P(condition ∧ parent_event)
                # intersection of condition and parent_event
                intersection = tuple(p_ae.intersect(condition) for p_ae in parent_event)
                intersection_prob = probability_of_conjunction(intersection)
                # Computing the probability of an empty intersection using the
                # chain rule may lead to undefined terms (if we condition
                # on an empty set).
                node_prob = node_prob + subject_given_condition * intersection_prob
                # All terms in the sum are divided by P(parent_event),
                # so we factor that out

            return node_prob / parents_prob

        return probability_of_conjunction(event)

    def density(self, elementary: torch.Tensor) -> torch.Tensor:
        # The probability density of an elementary event `e` is computed in the same
        # way as the probability of an event, but only one condition in a conditional
        # probability table can match for `e`, we compute the density in a single pass
        # over the network.
        # For the graph:
        # A     B
        #  ↘   ↙
        #    C
        #  ↙  ↘
        # D    E
        # p((a, b, c, d, e))
        # = p(E = e | A = a ∧ B = b ∧ C = c)
        #   * p(D = d | A = a ∧ B = b ∧ C = c)
        #   * p(C = c | A = a ∧ B = b)
        #   * p(A = a)
        #   * p(B = b)
        # where
        # p(E = e | A = a ∧ B = b ∧ C = c)
        # = p(E = e | matching_condition)
        #   * p(matching_condition ∧ A = a ∧ B = b ∧ C = c) / p(A = a ∧ B = b ∧ C = c)
        # = p(E = e | matching_condition)
        # where matching_condition is the condition in the conditional probability
        # table of E that matches for (a, b, c).
        # The last step in the derivation is since A = a ∧ B = b ∧ C = c implies
        # matching_condition and, therefore,
        # p(matching_condition ∧ A = a ∧ B = b ∧ C = c) = p(A = a ∧ B = b ∧ C = c).
        #
        # For hidden/latent variables, we sum out the values of the latent variable
        # as when computing probabilities.
        #
        if elementary.shape == self.event_shape:
            elementary = elementary.to(self.dtype).reshape(1, *self.event_shape)
        out_space = (
            self.__output_space
            if not self.include_hidden
            else self.__output_space_including_hidden
        )
        e = out_space.decompose(elementary)

        def condition_probability_hidden(hidden_node, condition):
            """
            Computes the probability that a hidden node matches a certain
            condition.
            """
            # Select the distribution that matches the elementary event.
            # We disallow other hidden nodes as parents of hidden nodes.
            for parent in hidden_node.parents:
                if parent.hidden:
                    raise NotImplementedError(
                        "Hidden nodes as parents of other hidden nodes are currently unsupported."
                        f"Node {hidden_node.name} as hidden parent {parent.name}."
                    )
            probability = torch.zeros(elementary.size(0), dtype=self.dtype)
            for cond, distr in hidden_node.conditional_probability_table:
                matches = torch.full((elementary.size(0),), fill_value=True)
                for parent, (l, u) in cond.items():
                    l = l.unsqueeze(0)  # add batch dim
                    u = u.unsqueeze(0)
                    parent_e = e[parent.name]
                    parent_matches = (l <= parent_e) & (parent_e <= u)
                    matches &= torch.all(parent_matches.flatten(1), dim=-1)
                probability += matches.float() * distr.probability(condition)
            return probability

        density = torch.ones(elementary.size(0), dtype=self.dtype)
        for node in self.__nodes:  # nodes are ordered from sources to sinks
            # Hidden nodes do not appear in the elementary event
            # and, therefore, do not influence the density.
            if not node.hidden or self.include_hidden:
                node_e = e[node.name]
                node_density = torch.zeros(elementary.size(0), dtype=self.dtype)
                for condition, distribution in node.conditional_probability_table:
                    conditional_density = distribution.density(node_e)
                    matches = torch.full((elementary.size(0),), fill_value=True)
                    for parent, (l, u) in condition.items():
                        l = l.unsqueeze(0)  # add batch dim
                        u = u.unsqueeze(0)
                        if not parent.hidden or self.include_hidden:
                            parent_e = e[parent.name]
                            parent_matches = (l <= parent_e) & (parent_e <= u)
                            matches &= torch.all(parent_matches.flatten(1), dim=-1)
                        else:
                            conditional_density *= condition_probability_hidden(
                                parent, (l, u)
                            )
                    node_density += matches.float() * conditional_density
                density *= node_density
        return density

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
            sample = torch.empty((num_samples, prod(self.__node_event_shapes[node])))
            for condition, distribution in node.conditional_probability_table:
                seed = seed_rng.randint(0, 2**64 - 1)
                sample_j = distribution.sample(num_samples, seed)
                sample_j = sample_j.reshape((num_samples, -1)).to(self.dtype)
                select = torch.full((num_samples,), fill_value=True)
                for parent in node.parents:
                    lower, upper = condition[parent]
                    select_k = (lower <= cache[parent]) & (cache[parent] <= upper)
                    select_k = torch.all(select_k.flatten(1), dim=1)
                    select &= select_k
                select.unsqueeze_(-1)
                sample = torch.where(select, sample_j, sample)
            cache[node] = sample
        return torch.hstack(
            [
                cache[node]
                for node in self.__original_order
                if self.include_hidden or not node.hidden
            ]
        )

    @property
    def include_hidden(self) -> bool:
        """
        Whether the events and samples of this Bayesian Network include hidden nodes.
        """
        return self.__include_hidden

    @include_hidden.setter
    def include_hidden(self, value: bool) -> None:
        self.__include_hidden = value

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size(self.__output_space.input_shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self.__dtype = dtype

    @property
    def parameters(self) -> torch.Tensor:
        return torch.hstack(
            [
                distribution.parameters.flatten()
                for node in self.__nodes
                for _, distribution in node.conditional_probability_table
            ]
        )

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        parameters = parameters.flatten()
        i = 0
        for node in self.__nodes:
            for _, distribution in node.conditional_probability_table:
                prev = distribution.parameters
                params = parameters[i : i + prev.numel()]
                distribution.parameters = params.reshape(prev.shape)
                i += prev.numel()

    @property
    def _parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        bounds = [
            distribution._parameter_bounds
            for node in self.__nodes
            for _, distribution in node.conditional_probability_table
        ]

        lbs, ubs = zip(*bounds)
        lbs = torch.hstack([lb.flatten() for lb in lbs])
        ubs = torch.hstack([ub.flatten() for ub in ubs])
        return lbs, ubs

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
                self.__conditional_probability_table: (
                    tuple[
                        tuple[
                            dict[str, tuple[torch.Tensor, torch.Tensor]],
                            ProbabilityDistribution,
                        ],
                        ...,
                    ]
                    | None
                ) = None
                self.__hidden: bool = False

            @property
            def name(self) -> str:
                return self.__name

            @property
            def hidden(self) -> bool:
                """
                Whether the node is a hidden (also known latent) node.
                """
                return self.__hidden

            @hidden.setter
            def hidden(self, hidden: bool):
                """
                Sets this node to be a hidden (or latent) node, or a regular node.
                """
                self.__hidden = hidden

            def add_parent(
                self,
                node: Union[str, "BayesianNetwork.Factory.Node"],
                reset_conditional_probabilities=False,
            ):
                """
                Add a parent node to this node.

                By default, the parents of a node can not be modified once
                the conditional probability table was created.
                Use :code:`reset_conditional_probabilities=True` to clear
                a previously set conditional probability table
                when modifying the parent nodes.
                """
                if reset_conditional_probabilities:
                    self.__conditional_probability_table = None
                else:
                    self._check_cond_prob_table_not_created("node parents")

                if not isinstance(node, str):
                    node = node.name
                self.__parents.append(node)

            def remove_parent(
                self,
                node: Union[str, "BayesianNetwork.Factory.Node"],
                reset_conditional_probabilities=False,
            ):
                """
                Remove a parent node of this node.

                By default, the parents of a node can not be modified once
                the conditional probability table was created.
                Use :code:`reset_conditional_probabilities=True` to clear
                a previously set conditional probability table
                when modifying the parent nodes.
                """
                if reset_conditional_probabilities:
                    self.__conditional_probability_table = None
                else:
                    self._check_cond_prob_table_not_created("node parents")

                if not isinstance(node, str):
                    node = node.name
                self.__parents.remove(node)

            def set_parents(
                self,
                *nodes: Union[str, "BayesianNetwork.Factory.Node"],
                reset_conditional_probabilities=False,
            ):
                """
                Set the parent nodes of this node.

                By default, the parents of a node can not be modified once
                the conditional probability table was created.
                Use :code:`reset_conditional_probabilities=True` to clear
                a previously set conditional probability table
                when modifying the parent nodes.
                """
                if reset_conditional_probabilities:
                    self.__conditional_probability_table = None
                else:
                    self._check_cond_prob_table_not_created("node parents")

                self.__parents = []
                for node in nodes:
                    self.add_parent(node)

            @property
            def parents(self) -> tuple[str, ...]:
                return tuple(self.__parents)

            def unbounded_event_space(
                self,
                reset_conditional_probabilities=False,
            ):
                """
                Sets this node to have an unbounded event space.

                By default, the event space of a node can not be modified once
                the conditional probability table was created.
                Use :code:`reset_conditional_probabilities=True` to clear
                a previously set conditional probability table when
                modifying the event space.
                """
                if reset_conditional_probabilities:
                    self.__conditional_probability_table = None
                else:
                    self._check_cond_prob_table_not_created("event space")
                self.__event_space = BayesianNetwork.Factory.EventSpace.Unbounded()

            def continuous_event_space(
                self,
                lower: TENSOR_LIKE,
                upper: TENSOR_LIKE,
                reset_conditional_probabilities=False,
            ):
                """
                Sets this node to have a continuous event space in the form
                of a hyper-rectangle with minimal elements :code:`lower`
                and maximal elements :code:`upper`.

                By default, the event space of a node can not be modified once
                the conditional probability table was created.
                Use :code:`reset_conditional_probabilities=True` to clear
                a previously set conditional probability table when
                modifying the event space.
                """
                if reset_conditional_probabilities:
                    self.__conditional_probability_table = None
                else:
                    self._check_cond_prob_table_not_created("event space")

                lower = to_tensor(lower)
                upper = to_tensor(upper)
                self.__event_space = BayesianNetwork.Factory.EventSpace.Continuous(
                    lower, upper
                )

            def discrete_event_space(
                self,
                *values: TENSOR_LIKE,
                reset_conditional_probabilities=False,
            ):
                """
                Sets this node to have a discrete event space with the
                given values as options.

                By default, the event space of a node can not be modified once
                the conditional probability table was created.
                Use :code:`reset_conditional_probabilities=True` to clear
                a previously set conditional probability table when
                modifying the event space.
                """
                if reset_conditional_probabilities:
                    self.__conditional_probability_table = None
                else:
                    self._check_cond_prob_table_not_created("event space")

                values = tuple(to_tensor(val) for val in values)
                self.__event_space = BayesianNetwork.Factory.EventSpace.Discrete(values)

            def one_hot_event_space(self, num_values: int):
                """
                Sets this node to have a discrete event space with
                :code:`num_values` one-hot encoded vectors.

                Concretely, when :code:`num_values=4` the event space consists of
                :code:`[1.0, 0.0, 0.0, 0.0] [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]`.

                :param num_values: The number of values of the one-hot encoded
                 variable.
                """
                self.discrete_event_space(*torch.eye(num_values).tolist())

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
                can not be changed without resetting the conditional probability
                table.

                All conditions in the conditional probability table need to be
                disjoint.

                :param condition: A mapping from parent node (names) to events.
                 Events can either be single values for discrete event spaces,
                 or tuples of minimal (lower) and maximal (upper) elements.
                 In this case, the minimal and maximal elements describe a rectangular
                 event (a set) in the event space of the parent node.
                 If the values produced by the parent nodes all lie in their respective
                 condition events, the value of this node is determined
                 by :code:`distribution`.
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

                # convert events to tensors and convert single value events to bounds
                for node, event in condition.items():
                    if isinstance(event, tuple):
                        lower, upper = event
                        lower = to_tensor(lower)
                        upper = to_tensor(upper)
                    else:
                        lower = upper = to_tensor(event)
                    condition[node] = (lower, upper)

                if self.__conditional_probability_table is not None:
                    self._check_event_shape(distribution)

                if self.__conditional_probability_table is None:
                    self.__conditional_probability_table = []
                self.__conditional_probability_table.append((condition, distribution))

            def _check_event_shape(self, distribution):
                """
                Check that a distribution has the same event shape as the other
                distributions in the conditional probability table.
                """
                _, other_distribution = self.__conditional_probability_table[0]
                if other_distribution.event_shape != distribution.event_shape:
                    raise ValueError(
                        f"Distribution {distribution} has a different event shape "
                        f"than previously supplied distribution "
                        f"{other_distribution}."
                    )

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
            self.__nodes: OrderedDict[str, "BayesianNetwork.Factory.Node"] = (
                OrderedDict()
            )
            self.dtype = torch.double

        def new_node(self, name: str, replace=False) -> "BayesianNetwork.Factory.Node":
            """
            Creates a new node with name :code:`name`.
            If a node with this name already exists and :code:`replace=True`,
            the existing node is replaced by a new node.
            Otherwise, if :code:`replace=True`, this method throws
            a :code:`ValueError`.
            """
            if name in self.__nodes and not replace:
                raise ValueError(
                    f"Node name {name} already used. Nodes need to have unique names."
                )
            node = BayesianNetwork.Factory.Node(name)
            self.__nodes[name] = node
            return node

        def new_nodes(
            self, *names: str, replace=False
        ) -> tuple["BayesianNetwork.Factory.Node", ...]:
            """
            Creates several new nodes.
            See :code:`add_node`.
            """
            return tuple(self.new_node(name, replace) for name in names)

        def node(self, name: str) -> "BayesianNetwork.Factory.Node":
            """
            Retrieves a node named :code:`name`.
            If no node with this name exists, a new node is created.
            """
            if name in self.__nodes:
                return self.__nodes[name]
            else:
                return self.new_node(name)

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
            for node in self.__nodes.values():
                if len(node.parents) == 0:
                    continue
                self._check_disjoint_conditions(node)
                self._check_parent_space_covered(node)

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
                            _ConditionalProbabilityTableEntry(
                                {
                                    nodes[p]: _Event(*event)
                                    for p, event in condition.items()
                                },
                                distribution,
                            )
                            for condition, distribution in node.conditional_probability_table
                        )
                        new_node = _Node(
                            node.name,
                            parents,
                            cond_prob_table,
                            node.hidden,
                        )
                        nodes[node.name] = new_node
                        processed.add(node.name)
            # recreate order from self.__nodes
            nodes = tuple(nodes[node_name] for node_name in self.__nodes)
            return BayesianNetwork(nodes, self.dtype)

        def _check_disjoint_conditions(self, node: "BayesianNetwork.Factory.Node"):
            """
            Checks if the conditions in the conditional probability table
            of a node are disjoint.
            """
            table = node.conditional_probability_table
            for i in range(len(table)):
                condition, _ = table[i]
                for j in range(i + 1, len(table)):
                    other_condition, _ = table[j]
                    any_disjoint = False
                    for parent, event in condition.items():
                        lower, upper = event
                        other_lower, other_upper = other_condition[parent]
                        # For continuous event spaces, intersections with measure
                        # zero are permissible, while not permissible for discrete
                        # event spaces.
                        match self.__nodes[parent].event_space:
                            case (
                                BayesianNetwork.Factory.EventSpace.Continuous(_, _)
                                | BayesianNetwork.Factory.EventSpace.Unbounded()
                            ):
                                if parent in other_condition:
                                    if torch.any(
                                        (lower > other_upper) | (upper < other_lower)
                                    ):
                                        any_disjoint = True
                                    if torch.all(
                                        (lower >= other_upper) | (upper <= other_lower)
                                    ):
                                        any_disjoint = True
                            case BayesianNetwork.Factory.EventSpace.Discrete(vals):
                                for val in vals:
                                    if torch.all(
                                        (lower <= val) & (val <= upper)
                                    ) and torch.all(
                                        (other_lower <= val) & (val <= other_upper)
                                    ):
                                        # not disjoint
                                        break
                                else:
                                    # no value lies in both event and other_condition
                                    any_disjoint = True
                        if any_disjoint:
                            break
                    if not any_disjoint:
                        raise ValueError(
                            f"Condition {condition} in conditional probability table of "
                            f"{node.name} not disjoint from other entry {other_condition}."
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
                            f"Condition {condition} in conditional probability table of "
                            f"{node.name} not identical to other entry {other_condition}."
                        )

        def _check_parent_space_covered(self, node: "BayesianNetwork.Factory.Node"):
            """
            Check whether the conditional probability table of a node
            covers the entire parent event space.
            """
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
                        return all(torch.all((lower <= v) & (v <= upper)) for v in vals)
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
                            case BayesianNetwork.Factory.EventSpace.Continuous(l, u):
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
                            contains(condition[p], new_part[p]) for p in node.parents
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
