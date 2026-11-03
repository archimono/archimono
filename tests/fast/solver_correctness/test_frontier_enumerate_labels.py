"""Regression test: frontier-DP enumerate_all emits a compact uint8 ndarray.

Before the refactor, ``FrontierExactSolver.solve(..., enumerate_all=True)``
stored ``all_labels`` as a list of 0/1 int tuples. It now stores a node-indexed
``uint8`` ndarray of shape ``(n_optima, n_nodes)`` and ``result.labels`` is one
of its rows. The compact representation packs straight into the certification
sidecar without an intermediate list of Python int-tuples.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from archimono.assignment.solvers.frontier_dp import FrontierExactSolver


def test_all_labels_are_uint8_ndarray() -> None:
    """metadata['all_labels'] is a uint8 (n_optima, 6) array; labels is one row."""
    graph = nx.cycle_graph(6)  # even cycle: balanced MAX-CUT has several optima
    result = FrontierExactSolver().solve(
        graph, ["A", "B"], n_b=3, enumerate_all=True
    )
    all_labels = result.metadata["all_labels"]
    assert isinstance(all_labels, np.ndarray)
    assert all_labels.dtype == np.uint8
    assert all_labels.ndim == 2
    assert all_labels.shape[0] >= 1  # at least one optimal assignment
    assert all_labels.shape[1] == graph.number_of_nodes()
    assert set(np.unique(all_labels).tolist()).issubset({0, 1})
    assert all(int(row.sum()) == 3 for row in all_labels)  # composition honoured
    # The representative is one of the enumerated optima.
    witness = tuple(int(x) for x in result.labels)
    assert witness in {tuple(int(x) for x in row) for row in all_labels}


def test_all_labels_absent_without_enumerate_all() -> None:
    graph = nx.cycle_graph(6)
    result = FrontierExactSolver().solve(graph, ["A", "B"], n_b=3)
    assert "all_labels" not in result.metadata


def test_max_optima_cap_raises_value_error() -> None:
    """enumerate_all=True with max_optima=1 raises on a multi-optima graph."""
    graph = nx.cycle_graph(6)
    with pytest.raises(ValueError, match="max_optima"):
        FrontierExactSolver().solve(
            graph, ["A", "B"], n_b=3, enumerate_all=True, max_optima=1
        )


def test_max_optima_cap_degenerate_graph() -> None:
    """Edgeless graph with exponential optima triggers the cap.

    An edgeless graph has C(n, n_b) optimal assignments (all compositions
    are optimal). empty_graph(10) with n_b=5 yields C(10, 5) = 252 optima,
    which exceeds max_optima=10.
    """
    graph = nx.empty_graph(10)
    with pytest.raises(ValueError, match="max_optima"):
        FrontierExactSolver().solve(
            graph, ["A", "B"], n_b=5, enumerate_all=True, max_optima=10
        )


def test_max_optima_sufficient_succeeds() -> None:
    """enumerate_all=True with a generous cap succeeds and returns all_labels."""
    graph = nx.cycle_graph(6)
    result = FrontierExactSolver().solve(
        graph, ["A", "B"], n_b=3, enumerate_all=True, max_optima=1000
    )
    all_labels = result.metadata["all_labels"]
    assert isinstance(all_labels, np.ndarray)
    assert all_labels.dtype == np.uint8
    assert all_labels.shape[0] >= 2
