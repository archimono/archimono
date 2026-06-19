"""Shared fixed-composition machinery for heuristic assignment solvers."""

from __future__ import annotations

import abc
from collections.abc import Sequence

from archimono.assignment.solvers import base

AdjacencyMap = dict[int, list[tuple[int, float]]]


class HeuristicAssignmentSolver(base.AssignmentSolver, abc.ABC):
    """Shared base for heuristic solvers with fixed-composition swap moves."""

    def _target_composition(
        self,
        n_nodes: int,
        n_b: int | None,
    ) -> base.TargetComposition:
        """Return the heuristic solver's fixed composition target."""
        if n_b is None:
            n_b = n_nodes // 2
        if not isinstance(n_b, int) or isinstance(n_b, bool):
            raise ValueError("n_b must be an integer count of 1 labels.")
        if n_b < 0 or n_b > n_nodes:
            raise ValueError(f"n_b must satisfy 0 <= n_b <= {n_nodes}.")
        return (n_nodes - n_b, n_b)

    @staticmethod
    def _adjacency_weights(
        edges: Sequence[base.AssignmentEdge],
    ) -> AdjacencyMap:
        """Build the weighted neighbor map used by the heuristic solvers.

        Args:
            edges: Weighted graph edges as `(i, j, weight)` triples.

        Returns:
            A dictionary keyed by node index. Each value is a list of
            `(neighbor_index, weight)` pairs for the non-loop edges incident to
            that node. Every undirected edge `(i, j, weight)` is stored twice:
            once under `i` as `(j, weight)` and once under `j` as `(i, weight)`.

        Notes:
            This adjacency structure is used only for local swap-delta
            calculations in the heuristic solvers. Self-loops are excluded
            because they never contribute to the cut and therefore should not
            appear in the local delta bookkeeping.
        """
        adjacency: AdjacencyMap = {}
        for i, j, weight in edges:
            if i == j:
                continue
            adjacency.setdefault(i, []).append((j, weight))
            adjacency.setdefault(j, []).append((i, weight))
        return adjacency

    @staticmethod
    def _initial_labels(target_composition: base.TargetComposition) -> list[int]:
        """Construct the deterministic initial `0/1` labels for a composition."""
        n_a, n_b = target_composition
        return [0] * n_a + [1] * n_b

    @staticmethod
    def _node_indices_by_label(labels: Sequence[int]) -> tuple[list[int], list[int]]:
        """Return node indices grouped by binary label."""
        one_nodes = [i for i, label in enumerate(labels) if label == 1]
        zero_nodes = [i for i, label in enumerate(labels) if label == 0]
        return one_nodes, zero_nodes

    @staticmethod
    def _swap_node_memberships(
        *,
        one_nodes: list[int],
        zero_nodes: list[int],
        one_index: int,
        zero_index: int,
        i: int,
        j: int,
    ) -> None:
        """Update cached label buckets after swapping one `1` node with one `0`.

        Args:
            one_nodes: Indices of nodes currently labeled `1`.
            zero_nodes: Indices of nodes currently labeled `0`.
            one_index: Position of `i` within `one_nodes`.
            zero_index: Position of `j` within `zero_nodes`.
            i: Node index currently labeled `1`.
            j: Node index currently labeled `0`.
        """
        one_nodes[one_index] = j
        zero_nodes[zero_index] = i

    @staticmethod
    def _swap_delta_cut_value(
        *,
        i: int,
        j: int,
        labels: Sequence[int],
        adjacency: AdjacencyMap,
    ) -> float:
        """Return the cut-value change induced by swapping two labels.

        Args:
            i: Index of one node to swap.
            j: Index of the other node to swap.
            labels: Current `0/1` node labels.
            adjacency: Weighted adjacency map used by the heuristic local-search
                solvers. This map contains only non-loop neighbors, so edges
                whose contributions are constant under label swaps, such as
                self-loops, are excluded from the delta calculation.

        Returns:
            The change in weighted cut value induced by swapping the labels at
            `i` and `j`. Positive values improve the objective.
        """
        li = labels[i]
        lj = labels[j]
        if li == lj:
            return 0.0

        delta = 0.0
        for neighbor, weight in adjacency.get(i, []):
            if neighbor == j:
                continue
            delta += weight if labels[neighbor] == li else -weight
        for neighbor, weight in adjacency.get(j, []):
            if neighbor == i:
                continue
            delta += weight if labels[neighbor] == lj else -weight
        return delta
