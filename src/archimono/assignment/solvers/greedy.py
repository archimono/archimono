"""Greedy local search for binary assignment problems.

Implements a deterministic-improvement solver for binary MAX-CUT under a
user-selected composition constraint. Each restart begins from a random
labeling with the requested number of `1` labels and repeatedly applies the
best improving swap between opposite-label nodes until no improving move
remains.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import networkx as nx

from archimono.assignment.solvers import (
    _heuristic_base,
    _scoring,
    _validation,
    base,
)


class GreedySolver(_heuristic_base.HeuristicAssignmentSolver):
    """Greedy local-search solver for composition-constrained binary MAX-CUT.

    The solver preserves composition by searching only over swaps between
    opposite-label nodes. Multiple randomized restarts are evaluated, and the
    best locally optimal assignment is returned. If `n_b` is omitted, the
    solver defaults to the current near-balanced target.
    """

    def __init__(
        self,
        *,
        n_restarts: int = 8,
        seed: int | None = None,
    ) -> None:
        """Initialize the greedy solver.

        Args:
            n_restarts: Number of randomized construction attempts to evaluate.
            seed: Optional base seed used to derive restart-specific RNG states.

        Raises:
            ValueError: If `n_restarts` is not positive.
        """
        if n_restarts <= 0:
            raise ValueError("n_restarts must be positive.")

        self.n_restarts = n_restarts
        self.seed = seed

    def _best_improving_swap(
        self,
        *,
        labels: Sequence[int],
        adjacency: _heuristic_base.AdjacencyMap,
        one_nodes: Sequence[int],
        zero_nodes: Sequence[int],
        rng: random.Random,
    ) -> tuple[int, int, int, int, float] | None:
        """Return the best cut-improving fixed-composition swap, if one exists.

        Args:
            labels: Current `0/1` node labels.
            adjacency: Weighted adjacency map of the assignment graph.
            one_nodes: Node indices currently labeled `1`.
            zero_nodes: Node indices currently labeled `0`.
            rng: Random generator used to shuffle tie-order traversal.

        Returns:
            A tuple describing the best improving swap, or `None` if no
            improving swap exists.
        """
        one_indices = list(range(len(one_nodes)))
        zero_indices = list(range(len(zero_nodes)))
        rng.shuffle(one_indices)
        rng.shuffle(zero_indices)

        best_swap: tuple[int, int, int, int, float] | None = None
        best_delta = 0.0

        for one_index in one_indices:
            i = one_nodes[one_index]
            for zero_index in zero_indices:
                j = zero_nodes[zero_index]
                delta = self._swap_delta_cut_value(
                    i=i,
                    j=j,
                    labels=labels,
                    adjacency=adjacency,
                )
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (one_index, zero_index, i, j, delta)

        return best_swap

    def _locally_optimize_labels(
        self,
        *,
        adjacency: _heuristic_base.AdjacencyMap,
        rng: random.Random,
        target_composition: base.TargetComposition,
    ) -> tuple[list[int], int]:
        """Optimize one random fixed-composition start via best-improving swaps.

        Args:
            adjacency: Weighted adjacency map of the assignment graph.
            rng: Random generator used to shuffle the initial assignment and
                tie-order traversal.
            target_composition: Required `(n_a, n_b)` composition target for the
                start and all successors.

        Returns:
            A tuple of the locally optimized labels and the number of improving
            swaps applied.
        """
        labels = self._initial_labels(target_composition)
        rng.shuffle(labels)
        improving_swaps = 0

        one_nodes, zero_nodes = self._node_indices_by_label(labels)

        while one_nodes and zero_nodes:
            best_swap = self._best_improving_swap(
                labels=labels,
                adjacency=adjacency,
                one_nodes=one_nodes,
                zero_nodes=zero_nodes,
                rng=rng,
            )
            if best_swap is None:
                break

            one_index, zero_index, i, j, _ = best_swap
            labels[i] = 0
            labels[j] = 1
            self._swap_node_memberships(
                one_nodes=one_nodes,
                zero_nodes=zero_nodes,
                one_index=one_index,
                zero_index=zero_index,
                i=i,
                j=j,
            )
            improving_swaps += 1

        return labels, improving_swaps

    def solve(
        self,
        graph: nx.Graph[int],
        species: Sequence[str],
        *,
        n_b: int | None = None,
    ) -> base.AssignmentResult:
        """Solve a binary assignment problem greedily.

        Args:
            graph: Assignment graph whose nodes are labeled by integer indices.
            species: Ordered species labels, such as `["A", "B"]`.
            n_b: Optional count of `1` labels. If omitted, initialize from the
                current near-balanced composition target.

        Returns:
            The best assignment found across all greedy restarts.

        Raises:
            ValueError: If the problem is not a non-empty two-species instance
                or if `n_b` lies outside the valid range.
            RuntimeError: If no assignment is produced.
        """

        _validation.validate_binary_species_problem(graph, species)

        edges = _scoring.weighted_edge_list(graph)
        n_nodes = graph.number_of_nodes()
        target_composition = self._target_composition(n_nodes, n_b)
        adjacency = self._adjacency_weights(edges)
        best_result: base.AssignmentResult | None = None

        for restart in range(self.n_restarts):
            rng = random.Random(None if self.seed is None else self.seed + restart)
            labels, improving_swaps = self._locally_optimize_labels(
                adjacency=adjacency,
                rng=rng,
                target_composition=target_composition,
            )

            cut_value = _scoring.maxcut_value(labels, edges)
            result = base.AssignmentResult(
                labels=tuple(labels),
                objective_value=cut_value,
                cut_value=cut_value,
                n_frustrated=sum(
                    1
                    for u, v in graph.edges()
                    if labels[int(u)] == labels[int(v)]
                ),
                solver=self.__class__.__name__,
                metadata={
                    "species": tuple(species),
                    "label_mapping": {0: species[0], 1: species[1]},
                    "target_composition": target_composition,
                    "restart": restart,
                    "n_restarts": self.n_restarts,
                    "n_improving_swaps": improving_swaps,
                },
            )

            if (
                best_result is None
                or result.objective_value > best_result.objective_value
            ):
                best_result = result

        if best_result is None:
            raise RuntimeError("GreedySolver did not produce any assignment.")

        return best_result
