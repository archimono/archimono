"""Contract tests for tiling geometry across all 11 tilings.

Verifies PBC validity (correct degree), node/edge counts at 2x2,
bipartiteness flag consistency, and connectivity.
"""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.tilings import registry

# Ground-truth properties per tiling:
# (key, n_vertices, coordination, is_bipartite, nodes_2x2, edges_2x2)
TILING_PROPERTIES = [
    ("hexagonal", 2, 3, True, 8, 12),
    ("square", 2, 4, True, 8, 16),
    ("truncated_square", 4, 3, True, 16, 24),
    ("truncated_trihexagonal", 12, 3, True, 48, 72),
    ("triangular", 1, 6, False, 4, 6),
    ("kagome", 3, 4, False, 12, 24),
    ("truncated_hexagonal", 6, 3, False, 24, 36),
    ("rhombitrihexagonal", 6, 4, False, 24, 48),
    ("snub_square", 4, 5, False, 16, 40),
    ("elongated_triangular", 4, 5, False, 16, 32),
    ("snub_hexagonal", 6, 5, False, 24, 60),
]


class TestTilingGraphCounts:
    """All 11 tilings produce correct node/edge counts at 2x2."""

    @pytest.mark.parametrize(
        "key,n_vertices,coordination,is_bipartite,nodes_2x2,edges_2x2",
        TILING_PROPERTIES,
        ids=[row[0] for row in TILING_PROPERTIES],
    )
    def test_2x2_node_edge_counts(
        self,
        key: str,
        n_vertices: int,
        coordination: int,
        is_bipartite: bool,
        nodes_2x2: int,
        edges_2x2: int,
    ) -> None:
        """Node and edge counts at 2x2 match documented values."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=(2, 2))
        assert graph.number_of_nodes() == nodes_2x2
        assert graph.number_of_edges() == edges_2x2


class TestPBCDegreeValidity:
    """Every node has the same degree (uniform PBC wrapping)."""

    @pytest.mark.parametrize(
        "key,n_vertices,coordination,is_bipartite,nodes_2x2,edges_2x2",
        TILING_PROPERTIES,
        ids=[row[0] for row in TILING_PROPERTIES],
    )
    def test_uniform_degree_at_2x2(
        self,
        key: str,
        n_vertices: int,
        coordination: int,
        is_bipartite: bool,
        nodes_2x2: int,
        edges_2x2: int,
    ) -> None:
        """All nodes have the same degree (PBC is symmetric)."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=(2, 2))
        expected_deg = 2 * edges_2x2 // nodes_2x2
        for node in graph.nodes():
            assert graph.degree(node) == expected_deg, (
                f"{key}: node {node} has degree {graph.degree(node)}, "
                f"expected {expected_deg}"
            )


class TestBipartitenessFlag:
    """is_bipartite property matches actual graph bipartiteness."""

    @pytest.mark.parametrize(
        "key",
        [row[0] for row in TILING_PROPERTIES],
    )
    def test_bipartite_flag_matches_graph(self, key: str) -> None:
        """Tiling.is_bipartite matches nx.is_bipartite(graph)."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=(2, 2))
        assert nx.is_bipartite(graph) == tiling.is_bipartite


class TestPbcFalsePlanarity:
    """pbc=False graphs are planar for all tilings at 2x2."""

    @pytest.mark.parametrize(
        "key",
        [row[0] for row in TILING_PROPERTIES],
    )
    def test_pbc_false_is_planar(self, key: str) -> None:
        """Removing PBC edges produces a planar graph."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=(2, 2), pbc=False)
        assert nx.check_planarity(graph)[0]


class TestGraphConnectivity:
    """All tiling graphs at 2x2 are connected (single component)."""

    def test_all_tilings_connected(self, all_tilings: str) -> None:
        """Every tiling at 2x2 produces a connected graph."""
        tiling = registry.get(all_tilings)
        graph = tiling.graph(supercell=(2, 2))
        assert nx.is_connected(graph)
