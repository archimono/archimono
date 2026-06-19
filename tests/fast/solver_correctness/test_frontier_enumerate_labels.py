"""Regression test: frontier-DP enumerate_all emits 0/1 int labels (audit L5).

Before the fix, ``FrontierExactSolver.solve(..., enumerate_all=True)`` stored
``all_labels`` as tuples of species *strings*, which crashed the certification
sidecar packer (``int(v)`` on ``"A"``/``"B"``). The fix stores 0/1 int tuples,
matching ``labels`` and every consumer.
"""

from __future__ import annotations

import networkx as nx

from archimono.assignment.solvers.frontier_dp import FrontierExactSolver


def test_all_labels_are_int_tuples() -> None:
    """metadata['all_labels'] are 0/1 int tuples of length n, not species strings."""
    graph = nx.cycle_graph(6)  # even cycle: balanced MAX-CUT has several optima
    result = FrontierExactSolver().solve(
        graph, ["A", "B"], n_b=3, enumerate_all=True
    )
    all_labels = result.metadata["all_labels"]
    assert all_labels, "expected at least one optimal assignment"
    for labelling in all_labels:
        assert len(labelling) == graph.number_of_nodes()
        assert all(isinstance(v, int) and v in (0, 1) for v in labelling)
        assert sum(labelling) == 3  # composition honoured
    # The representative is one of the enumerated optima.
    assert tuple(result.labels) in {tuple(x) for x in all_labels}


def test_all_labels_absent_without_enumerate_all() -> None:
    """Default solve does not populate all_labels."""
    graph = nx.cycle_graph(6)
    result = FrontierExactSolver().solve(graph, ["A", "B"], n_b=3)
    assert "all_labels" not in result.metadata
