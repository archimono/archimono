"""Generic graph and objective helpers for assignment solvers."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from archimono.assignment.solvers import _validation, base


def weighted_edge_list(graph: nx.Graph[int]) -> list[base.AssignmentEdge]:
    """Return graph edges as `(i, j, weight)` triples.

    Args:
        graph: Graph whose edges may include optional `weight` attributes.

    Returns:
        A list of weighted edge triples with integer node indices.
    """
    edges: list[base.AssignmentEdge] = []
    for i, j, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        edges.append((int(i), int(j), weight))
    return edges


def maxcut_value(
    labels: Sequence[int],
    edges: Sequence[base.AssignmentEdge],
) -> float:
    """Return the weighted cut value for a binary assignment.

    Args:
        labels: Node labels encoded as `0` and `1`, with opposite labels
            representing opposite sides of the cut.
        edges: Weighted graph edges as `(i, j, weight)` triples.

    Returns:
        The weighted cut value induced by `labels`.

    Raises:
        ValueError: If `labels` contains values other than `0` and `1`.
    """
    _validation.validate_labels(labels)

    cut_value = 0.0
    for i, j, weight in edges:
        cut_value += weight if labels[i] != labels[j] else 0.0
    return cut_value
