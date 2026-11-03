"""Validation helpers shared across assignment solvers."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

_VALID_LABELS = (0, 1)


def validate_binary_species_problem(
    graph: nx.Graph[int],
    species: Sequence[str],
) -> None:
    """Validate that the inputs define a supported binary assignment problem.

    Raises:
        ValueError: If the problem is not a non-empty two-species assignment.
    """
    if len(species) != 2:
        raise ValueError("Binary assignment solvers currently support two species.")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node.")
    validate_node_labeled_graph(graph)


def validate_node_labeled_graph(graph: nx.Graph[int]) -> None:
    """Validate that graph nodes are indexed as `0 .. n - 1`.

    Raises:
        ValueError: If node labels are not contiguous integers.
    """
    nodes = sorted(int(node) for node in graph.nodes())
    expected = list(range(graph.number_of_nodes()))
    if nodes != expected:
        raise ValueError("graph nodes must be labeled by contiguous integers 0..n-1.")


def resolve_fixed_composition(n_nodes: int, n_b: int | None) -> tuple[int, int]:
    """Return the exact ``(n_a, n_b)`` composition target, defaulting ``n_b``.

    Args:
        n_nodes: Total number of graph nodes.
        n_b: Requested count of ``1`` labels, or ``None`` to default to
            ``n_nodes // 2``.

    Raises:
        ValueError: If ``n_b`` is not a valid integer in ``[0, n_nodes]``.
    """
    if n_b is None:
        n_b = n_nodes // 2
    if not isinstance(n_b, int) or isinstance(n_b, bool):
        raise ValueError("n_b must be an integer count of 1 labels.")
    if n_b < 0 or n_b > n_nodes:
        raise ValueError(f"n_b must satisfy 0 <= n_b <= {n_nodes}.")
    return (n_nodes - n_b, n_b)


def validate_labels(labels: Sequence[int]) -> None:
    """Validate that a label sequence uses the expected binary encoding."""
    if any(label not in _VALID_LABELS for label in labels):
        raise ValueError("labels must be a sequence of 0/1 assignment labels.")
