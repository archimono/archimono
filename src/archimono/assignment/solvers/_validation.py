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

    Args:
        graph: Assignment graph to solve.
        species: Ordered species labels expected by the solver.

    Raises:
        ValueError: If the problem is not a non-empty two-species assignment.
    """
    if len(species) != 2:
        raise ValueError("Binary assignment solvers currently support two species.")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node.")
    validate_node_labeled_graph(graph)


def validate_node_labeled_graph(graph: nx.Graph[int]) -> None:
    """Validate the node-labeling assumptions used by assignment solvers.

    Args:
        graph: Assignment graph whose nodes should be indexed as `0 .. n - 1`.

    Raises:
        ValueError: If node labels are not contiguous integers.
    """
    nodes = sorted(int(node) for node in graph.nodes())
    expected = list(range(graph.number_of_nodes()))
    if nodes != expected:
        raise ValueError("graph nodes must be labeled by contiguous integers 0..n-1.")


def validate_labels(labels: Sequence[int]) -> None:
    """Validate that a label sequence uses the expected binary encoding."""
    if any(label not in _VALID_LABELS for label in labels):
        raise ValueError("labels must be a sequence of 0/1 assignment labels.")
