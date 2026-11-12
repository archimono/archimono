"""Unit tests for the frontier-DP reconstruction memory estimator."""

from __future__ import annotations

import networkx as nx
import pytest

import archimono.assignment as assignment
from archimono.assignment.solvers import frontier_dp
from archimono.assignment.solvers.frontier_dp import (
    _best_geometric_order,
    compute_dp_cost,
    estimate_reconstruct_memory_bytes,
)


def test_reconstruct_matches_formula_for_representative_dp_cost_and_n() -> None:
    """Estimate equals int(2**3.9 * dp_cost**0.99 * n**1.2 * 1.5)."""
    for dp_cost, n in [(12.0, 12), (1024.0, 24), (262144.0, 40)]:
        expected = int(2**3.9 * dp_cost**0.99 * n**1.2 * 1.5)
        assert estimate_reconstruct_memory_bytes(dp_cost, n) == expected


def test_reconstruct_min_dp_cost_is_positive() -> None:
    """The no-edge lower bound (dp_cost=n, here dp_cost=1, n=1) stays positive."""
    assert estimate_reconstruct_memory_bytes(1.0, 1) > 0


def test_reconstruct_estimator_is_exported() -> None:
    assert (
        assignment.estimate_reconstruct_memory_bytes
        is frontier_dp.estimate_reconstruct_memory_bytes
    )


def test_parent_maps_estimator_is_removed() -> None:
    """The replaced estimator is gone from the module and the package."""
    assert not hasattr(frontier_dp, "estimate_parent_maps_bytes")
    assert not hasattr(assignment, "estimate_parent_maps_bytes")


def test_compute_dp_cost_geometric_matches_best_order() -> None:
    """compute_dp_cost('geometric') equals _best_geometric_order(graph).dp_cost."""
    graph: nx.Graph[int] = nx.path_graph(6)
    assert (
        compute_dp_cost(graph, "geometric") == _best_geometric_order(graph).dp_cost
    )


def test_compute_dp_cost_unknown_heuristic_raises() -> None:
    graph: nx.Graph[int] = nx.path_graph(4)
    with pytest.raises(ValueError):
        compute_dp_cost(graph, "bogus")


def test_compute_dp_cost_is_exported() -> None:
    assert assignment.compute_dp_cost is frontier_dp.compute_dp_cost
