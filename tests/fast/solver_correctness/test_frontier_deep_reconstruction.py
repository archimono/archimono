"""Regression test: frontier-DP label reconstruction survives deep graphs.

``_enumerate_from_parent_dag`` used to recurse once per vertex, so any
label-returning frontier solve on a graph with more nodes than the Python
recursion limit (~1000) crashed with ``RecursionError`` after the forward DP
had already succeeded. The traversal must not depend on graph size.
"""

from __future__ import annotations

import networkx as nx

from archimono.assignment.solvers.frontier_dp import FrontierExactSolver


def test_solve_returns_labels_on_deep_path_graph() -> None:
    """A 1500-node path solve returns the alternating optimum, not RecursionError."""
    graph = nx.path_graph(1500)
    result = FrontierExactSolver().solve(graph, ["A", "B"], n_b=750)
    assert result.cut_value == 1499.0
    assert result.n_frustrated == 0
    assert sum(result.labels) == 750


def test_enumerate_all_on_deep_path_graph() -> None:
    """enumerate_all walks the parent DAG iteratively on a 1500-node path."""
    graph = nx.path_graph(1500)
    result = FrontierExactSolver().solve(
        graph, ["A", "B"], n_b=750, enumerate_all=True
    )
    all_labels = result.metadata["all_labels"]
    # The alternating 2-coloring and its complement are the only optima.
    assert all_labels.shape == (2, 1500)
