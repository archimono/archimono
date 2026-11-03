"""Tests for the reachability-aware value-only memory estimator.

Covers the ``compute_frontier_sizes`` accessor and the reachability-aware
``estimate_peak_memory_bytes`` formula that replaces the blanket
``2^k * (n // 2 + 1)`` peak-state bound (issue #77 §2).
"""

from __future__ import annotations

import networkx as nx
import pytest

import archimono.assignment as assignment
from archimono.assignment.solvers import frontier_dp
from archimono.assignment.solvers.frontier_dp import (
    _best_geometric_order,
    _solve_frontier_dp,
    compute_dp_cost,
    compute_frontier_sizes,
    compute_frontier_width,
    estimate_peak_memory_bytes,
)
from archimono.tilings import registry

_HEURISTICS = ("natural", "ck", "greedy", "geometric")

_BYTES_PER_ENTRY = 160


def test_frontier_sizes_has_one_entry_per_step_and_max_equals_width() -> None:
    graph: nx.Graph[int] = nx.path_graph(6)
    for heuristic in _HEURISTICS:
        sizes = compute_frontier_sizes(graph, heuristic)
        assert len(sizes) == graph.number_of_nodes()
        assert max(sizes) == compute_frontier_width(graph, heuristic)


def test_frontier_sizes_consistent_with_dp_cost() -> None:
    """sum(2**size) over the sequence equals compute_dp_cost."""
    graph: nx.Graph[int] = nx.cycle_graph(8)
    for heuristic in _HEURISTICS:
        sizes = compute_frontier_sizes(graph, heuristic)
        assert sum(2.0**s for s in sizes) == compute_dp_cost(graph, heuristic)


def test_frontier_sizes_unknown_heuristic_raises() -> None:
    graph: nx.Graph[int] = nx.path_graph(4)
    with pytest.raises(ValueError):
        compute_frontier_sizes(graph, "bogus")


def test_frontier_sizes_is_exported() -> None:
    assert (
        assignment.compute_frontier_sizes is frontier_dp.compute_frontier_sizes
    )


def test_estimate_matches_hand_computed_reachable_counts() -> None:
    """Path graph 0-1-2, n_b=1: reachable layer counts are (2, 3, 2).

    Frontier sizes for the natural order are (1, 2, 1).  Per layer t the
    reachable count is sum over popcount p of C(width, p) * |valid n_ones
    range|, giving 2, 3, 2 — so the estimate is 2 * max(3) * 160 = 960.
    """
    assert estimate_peak_memory_bytes((1, 2, 1), 1) == 2 * 3 * _BYTES_PER_ENTRY


def test_estimate_never_underestimates_actual_retained_peak() -> None:
    """Estimated states cover the DP's measured max_states_retained.

    The reachable-state count is an upper bound on every retained layer, so
    ``estimate / (2 * 160)`` must be >= the actual peak layer size reported
    by a real value-only solve on the same (geometric) ordering.
    """
    graphs: list[nx.Graph[int]] = [
        nx.path_graph(6),
        nx.cycle_graph(6),
        nx.complete_graph(6),
        registry.get("kagome").graph((2, 2)),
        registry.get("triangular").graph((3, 3)),
        registry.get("square").graph((3, 3)),
    ]
    for graph in graphs:
        n_b = graph.number_of_nodes() // 2
        sizes = compute_frontier_sizes(graph, "geometric")
        estimate = estimate_peak_memory_bytes(sizes, n_b)
        _, _, meta = _solve_frontier_dp(
            graph=graph,
            order=_best_geometric_order(graph).order,
            n_b=n_b,
            value_only=True,
        )
        actual_peak_states = meta["max_states_retained"]
        assert estimate >= 2 * _BYTES_PER_ENTRY * actual_peak_states


def test_estimate_never_exceeds_blanket_bound() -> None:
    """The reachability-aware estimate is <= the old blanket formula.

    The blanket bound assumed every bitmask co-occurs with every n_ones in
    ``[0, n//2]``; the reachability count only tightens it, so the new
    estimate can never newly reject a case the old formula admitted.
    """
    graphs: list[nx.Graph[int]] = [
        nx.path_graph(6),
        registry.get("kagome").graph((2, 2)),
        registry.get("triangular").graph((3, 3)),
    ]
    for graph in graphs:
        n = graph.number_of_nodes()
        n_b = n // 2
        sizes = compute_frontier_sizes(graph, "geometric")
        k = max(sizes)
        blanket = 2 * (1 << k) * (n // 2 + 1) * _BYTES_PER_ENTRY
        assert estimate_peak_memory_bytes(sizes, n_b) <= blanket


def test_estimate_positive_for_edgeless_graph() -> None:
    """A graph with no edges (all frontier sizes 0) still estimates > 0."""
    graph: nx.Graph[int] = nx.empty_graph(4)
    sizes = compute_frontier_sizes(graph, "natural")
    assert sizes == (0, 0, 0, 0)
    assert estimate_peak_memory_bytes(sizes, 2) > 0
