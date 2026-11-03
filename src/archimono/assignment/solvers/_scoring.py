"""Generic graph and objective helpers for assignment solvers."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from archimono.assignment.solvers import _validation, base


def weighted_edge_list(graph: nx.Graph[int]) -> list[base.AssignmentEdge]:
    """Return graph edges as `(i, j, weight)` triples with integer node indices."""
    edges: list[base.AssignmentEdge] = []
    for i, j, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        edges.append((int(i), int(j), weight))
    return edges


def count_cut_edges(graph: nx.Graph[int], labels: Sequence[int]) -> int:
    """Count unit-weight cut edges (endpoints with different labels)."""
    return sum(1 for u, v in graph.edges() if labels[int(u)] != labels[int(v)])


def count_frustrated_edges(graph: nx.Graph[int], labels: Sequence[int]) -> int:
    """Count frustrated edges (endpoints with identical labels)."""
    return sum(1 for u, v in graph.edges() if labels[int(u)] == labels[int(v)])


def maxcut_value(
    labels: Sequence[int],
    edges: Sequence[base.AssignmentEdge],
) -> float:
    """Return the weighted cut value for a binary assignment.

    Raises:
        ValueError: If `labels` contains values other than `0` and `1`.
    """
    _validation.validate_labels(labels)

    cut_value = 0.0
    for i, j, weight in edges:
        cut_value += weight if labels[i] != labels[j] else 0.0
    return cut_value
