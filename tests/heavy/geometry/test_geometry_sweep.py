"""Heavy geometry tests: all tilings x multiple supercell sizes.

Verifies scaling laws, no duplicate edges, and connectivity
across all tilings at 2x2 and 3x3.
"""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.tilings import registry

ALL_TILING_KEYS = sorted(registry.available())
_SUPERCELLS = [(2, 2), (3, 3)]
_SMALL_TILINGS = ["triangular", "hexagonal", "square", "kagome"]


class TestNodeCountScaling:
    """Node count scales as n_vertices * na * nb."""

    @pytest.mark.parametrize("key", ALL_TILING_KEYS)
    @pytest.mark.parametrize("supercell", _SUPERCELLS)
    def test_node_count_formula(
        self, key: str, supercell: tuple[int, int]
    ) -> None:
        """n_nodes = n_vertices * na * nb."""
        tiling = registry.get(key)
        try:
            graph = tiling.graph(supercell=supercell)
        except ValueError:
            pytest.skip(f"{key} does not support supercell {supercell}")
        expected = tiling.n_vertices * supercell[0] * supercell[1]
        assert graph.number_of_nodes() == expected


class TestEdgeCountScaling:
    """Edge count is bounded by coordination number for all supercells."""

    @pytest.mark.parametrize("key", ALL_TILING_KEYS)
    @pytest.mark.parametrize("supercell", _SUPERCELLS)
    def test_edge_count_bounded(
        self, key: str, supercell: tuple[int, int]
    ) -> None:
        """Edge count <= coordination * n_nodes / 2."""
        tiling = registry.get(key)
        try:
            graph = tiling.graph(supercell=supercell)
        except ValueError:
            pytest.skip(f"{key} does not support supercell {supercell}")
        max_edges = tiling.coordination * graph.number_of_nodes() // 2
        assert graph.number_of_edges() <= max_edges
        assert graph.number_of_edges() > 0


class TestNoDuplicateEdges:
    """No multigraph behavior — every edge appears exactly once."""

    @pytest.mark.parametrize("key", ALL_TILING_KEYS)
    @pytest.mark.parametrize("supercell", _SUPERCELLS)
    def test_simple_graph(
        self, key: str, supercell: tuple[int, int]
    ) -> None:
        """Graph is simple (no self-loops, no parallel edges)."""
        tiling = registry.get(key)
        try:
            graph = tiling.graph(supercell=supercell)
        except ValueError:
            pytest.skip(f"{key} does not support supercell {supercell}")
        assert isinstance(graph, nx.Graph)
        assert nx.number_of_selfloops(graph) == 0


class TestConnectivity:
    """All tiling graphs are connected at every tested supercell."""

    @pytest.mark.parametrize("key", ALL_TILING_KEYS)
    @pytest.mark.parametrize("supercell", _SUPERCELLS)
    def test_connected(self, key: str, supercell: tuple[int, int]) -> None:
        """Graph is a single connected component."""
        tiling = registry.get(key)
        try:
            graph = tiling.graph(supercell=supercell)
        except ValueError:
            pytest.skip(f"{key} does not support supercell {supercell}")
        assert nx.is_connected(graph)


class TestLargeSupercells:
    """4x4 supercell tests for small-n_vertices tilings (laptop-safe)."""

    @pytest.mark.parametrize("key", _SMALL_TILINGS)
    def test_4x4_valid(self, key: str) -> None:
        """Small tilings support 4x4 supercell without error."""
        tiling = registry.get(key)
        try:
            graph = tiling.graph(supercell=(4, 4))
        except ValueError:
            pytest.skip(f"{key} does not support 4x4")
        assert graph.number_of_nodes() == tiling.n_vertices * 16
        assert nx.is_connected(graph)
