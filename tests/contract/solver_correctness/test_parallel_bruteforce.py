"""Contract tests for parallel fixed-composition brute force.

The parallel solver must reproduce the monolithic serial :class:`BruteforceSolver`
exactly. That is the correctness contract: no candidate may be missed, and the
returned optimum and optimal labels must match the serial solver.
"""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.assignment import BruteforceSolver
from archimono.assignment.solvers import _scoring
from archimono.assignment.solvers.parallel_bruteforce import (
    solve_fixed_composition_parallel,
)
from archimono.tilings import registry


def _labels_to_set(labels_iterable: object) -> set[tuple[int, ...]]:
    """Normalize an iterable of label tuples into a set for comparison."""
    assert isinstance(labels_iterable, list)
    return {tuple(int(x) for x in lbl) for lbl in labels_iterable}


def _bruteforce_ground_truth(
    graph: nx.Graph[int], n_b: int
) -> tuple[float, set[tuple[int, ...]]]:
    """Return the exact max cut and the full set of optimal labelings."""
    result = BruteforceSolver().solve(graph, ["A", "B"], n_b=n_b)
    assert result.cut_value is not None
    optimal = _labels_to_set(result.metadata["optimal_labels"])
    return result.cut_value, optimal


def _hexagonal_2x2() -> nx.Graph[int]:
    return registry.get("hexagonal").graph((2, 2))


def _c5() -> nx.Graph[int]:
    return nx.cycle_graph(5)


_GRAPH_CASES = [
    pytest.param(_hexagonal_2x2, 4, id="hexagonal-2x2-nb4"),
    pytest.param(_c5, 2, id="c5-nb2"),
]


class TestParallelBruteforceMatchesMonolithic:
    """The parallel solver reproduces the serial brute-force result exactly."""

    @pytest.mark.parametrize(("build_graph", "n_b"), _GRAPH_CASES)
    def test_parallel_equals_bruteforce(
        self, build_graph: object, n_b: int
    ) -> None:
        assert callable(build_graph)
        graph = build_graph()
        edges = _scoring.weighted_edge_list(graph)
        n = graph.number_of_nodes()
        target_composition = (n - n_b, n_b)
        gt_cut, gt_labels = _bruteforce_ground_truth(graph, n_b)

        labels, cut_value, metadata = solve_fixed_composition_parallel(
            edges=edges, graph=graph, target_composition=target_composition
        )

        assert cut_value == pytest.approx(gt_cut)
        assert _labels_to_set(metadata["optimal_labels"]) == gt_labels
        assert tuple(labels) in gt_labels
        assert labels == min(metadata["optimal_labels"])

    @pytest.mark.parametrize(("build_graph", "n_b"), _GRAPH_CASES)
    def test_parallel_without_numba(
        self, monkeypatch: pytest.MonkeyPatch, build_graph: object, n_b: int
    ) -> None:
        """Parallel solver matches serial even when Numba is disabled."""
        from archimono.assignment.solvers import parallel_bruteforce
        monkeypatch.setattr(parallel_bruteforce._jit, "HAS_NUMBA", False)

        assert callable(build_graph)
        graph = build_graph()
        edges = _scoring.weighted_edge_list(graph)
        n = graph.number_of_nodes()
        target_composition = (n - n_b, n_b)
        gt_cut, gt_labels = _bruteforce_ground_truth(graph, n_b)

        import io
        stream = io.StringIO()
        labels, cut_value, metadata = solve_fixed_composition_parallel(
            edges=edges,
            graph=graph,
            target_composition=target_composition,
            stream=stream,
        )

        assert cut_value == pytest.approx(gt_cut)
        assert _labels_to_set(metadata["optimal_labels"]) == gt_labels
        assert "numba is absent; falling back to serial bruteforce" in stream.getvalue()


class TestNearTieFloatWeights:
    """Float weights spaced within tolerance must not leak spurious optima."""

    @staticmethod
    def _near_tie_graph() -> nx.Graph[int]:
        """A 7-node weighted graph whose top cut values differ within 1e-9."""
        edges = [
            (0, 1, 1.0000000005), (0, 2, 1.0000000005), (0, 3, 2.0),
            (0, 4, 2.0), (0, 5, 1.0), (1, 3, 2.0), (1, 4, 2.0),
            (1, 5, -1.0), (2, 3, 2.0), (2, 4, 1.0), (2, 6, -1.0),
            (3, 4, 1.0), (3, 5, 1.0), (3, 6, -1.0), (4, 5, 1.0000000005),
            (5, 6, 2.0),
        ]
        graph: nx.Graph[int] = nx.Graph()
        graph.add_nodes_from(range(7))
        for i, j, w in edges:
            graph.add_edge(i, j, weight=w)
        return graph

    def test_no_spurious_optima_under_near_ties(self) -> None:
        """Parallel solver matches serial even under near-tie weights."""
        graph = self._near_tie_graph()
        edges = _scoring.weighted_edge_list(graph)
        n = graph.number_of_nodes()
        gt_cut, gt_labels = _bruteforce_ground_truth(graph, 1)

        labels, cut_value, metadata = solve_fixed_composition_parallel(
            edges=edges, graph=graph, target_composition=(n - 1, 1)
        )

        assert cut_value == pytest.approx(gt_cut)
        assert _labels_to_set(metadata["optimal_labels"]) == gt_labels
