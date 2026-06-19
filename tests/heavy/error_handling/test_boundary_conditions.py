"""Heavy error handling: boundary conditions at scale and malformed inputs.

Tests that validation holds on larger inputs and rejects malformed graphs.
"""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.assignment import AnnealingSolver, GreedySolver
from archimono.tilings import registry

ALL_TILING_KEYS = sorted(registry.available())


class TestLargestValidInputs:
    """Heuristic solvers handle the largest inputs just below feasibility limits."""

    @pytest.mark.parametrize("key", ALL_TILING_KEYS)
    def test_greedy_3x3_does_not_crash(self, key: str) -> None:
        """GreedySolver handles a 3x3 supercell without crashing."""
        tiling = registry.get(key)
        try:
            graph = tiling.graph(supercell=(3, 3))
        except ValueError:
            pytest.skip(f"{key} does not support 3x3")

        n = graph.number_of_nodes()
        n_b = n // 2

        result = GreedySolver(seed=42, n_restarts=3).solve(
            graph, ["A", "B"], n_b=n_b
        )
        assert result.cut_value >= 0
        assert len(result.labels) == n

    @pytest.mark.parametrize("key", ALL_TILING_KEYS)
    def test_annealing_3x3_does_not_crash(self, key: str) -> None:
        """AnnealingSolver handles a 3x3 supercell without crashing."""
        tiling = registry.get(key)
        try:
            graph = tiling.graph(supercell=(3, 3))
        except ValueError:
            pytest.skip(f"{key} does not support 3x3")

        n = graph.number_of_nodes()
        n_b = n // 2

        result = AnnealingSolver(seed=42, n_restarts=3).solve(
            graph, ["A", "B"], n_b=n_b
        )
        assert result.cut_value >= 0
        assert len(result.labels) == n


class TestMalformedGraphs:
    """Solvers handle or reject malformed graph inputs."""

    def test_disconnected_graph(self) -> None:
        """Solver handles disconnected graph without crash."""
        graph: nx.Graph[int] = nx.Graph()
        graph.add_edges_from([(0, 1), (2, 3)])
        result = GreedySolver(seed=42).solve(graph, ["A", "B"])
        assert result.cut_value >= 0
        assert len(result.labels) == 4
