"""Contract tests for tiling geometry across all 11 tilings.

Verifies PBC validity (correct degree), node/edge counts, bipartiteness
flag consistency, and connectivity, each at the smallest supercell size
supported by the current simple-graph representation for that tiling.
"""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.tilings import Tiling, registry

# Ground-truth properties per tiling:
# (key, n_vertices, coordination, is_bipartite, nodes, edges) at the
# supercell returned by _supercell_for(tiling) below.
TILING_PROPERTIES = [
    ("hexagonal", 2, 3, True, 8, 12),
    ("square", 2, 4, True, 8, 16),
    ("truncated_square", 4, 3, True, 16, 24),
    ("truncated_trihexagonal", 12, 3, True, 48, 72),
    ("triangular", 1, 6, False, 9, 27),
    ("kagome", 3, 4, False, 12, 24),
    ("truncated_hexagonal", 6, 3, False, 24, 36),
    ("rhombitrihexagonal", 6, 4, False, 24, 48),
    ("snub_square", 4, 5, False, 16, 40),
    ("elongated_triangular", 4, 5, False, 12, 30),
    ("snub_hexagonal", 6, 5, False, 24, 60),
]

# triangular and elongated_triangular alias at (2,2) (edge instances collapse
# under PBC -- see Tiling._graph_matrix); their smallest supported supercells
# are (3,3) and (3,1) respectively. Every other tiling is unaffected at
# (2,2). Keyed by vertex_config (not the registry lookup key) since the
# all_tilings fixture parametrizes over every registered alias, including
# vertex-config strings, for the same tiling.
_SUPERCELL_BY_VERTEX_CONFIG: dict[str, tuple[int, int]] = {
    "3⁶": (3, 3),  # triangular
    "3³.4²": (3, 1),  # elongated_triangular
}
_DEFAULT_SUPERCELL: tuple[int, int] = (2, 2)


def _supercell_for(tiling: Tiling) -> tuple[int, int]:
    """Return the smallest supported supercell for *tiling*."""
    return _SUPERCELL_BY_VERTEX_CONFIG.get(tiling.vertex_config, _DEFAULT_SUPERCELL)


class TestTilingGraphCounts:
    """All 11 tilings produce correct node/edge counts at their supported size."""

    @pytest.mark.parametrize(
        "key,n_vertices,coordination,is_bipartite,nodes,edges",
        TILING_PROPERTIES,
        ids=[row[0] for row in TILING_PROPERTIES],
    )
    def test_node_edge_counts(
        self,
        key: str,
        n_vertices: int,
        coordination: int,
        is_bipartite: bool,
        nodes: int,
        edges: int,
    ) -> None:
        """Node and edge counts at the supported supercell match documented values."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=_supercell_for(tiling))
        assert graph.number_of_nodes() == nodes
        assert graph.number_of_edges() == edges


class TestPBCDegreeValidity:
    """Every node has the same degree (uniform PBC wrapping)."""

    @pytest.mark.parametrize(
        "key,n_vertices,coordination,is_bipartite,nodes,edges",
        TILING_PROPERTIES,
        ids=[row[0] for row in TILING_PROPERTIES],
    )
    def test_uniform_degree(
        self,
        key: str,
        n_vertices: int,
        coordination: int,
        is_bipartite: bool,
        nodes: int,
        edges: int,
    ) -> None:
        """All nodes have the same degree (PBC is symmetric)."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=_supercell_for(tiling))
        expected_deg = 2 * edges // nodes
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
        tiling = registry.get(key)
        graph = tiling.graph(supercell=_supercell_for(tiling))
        assert nx.is_bipartite(graph) == tiling.is_bipartite


class TestPbcFalsePlanarity:
    """pbc=False graphs are planar for all tilings at their supported size."""

    @pytest.mark.parametrize(
        "key",
        [row[0] for row in TILING_PROPERTIES],
    )
    def test_pbc_false_is_planar(self, key: str) -> None:
        tiling = registry.get(key)
        graph = tiling.graph(supercell=_supercell_for(tiling), pbc=False)
        assert nx.check_planarity(graph)[0]


class TestGraphConnectivity:
    """All tiling graphs are connected (single component) at their supported size."""

    def test_all_tilings_connected(self, all_tilings: str) -> None:
        """Every tiling produces a connected graph at its supported supercell."""
        tiling = registry.get(all_tilings)
        graph = tiling.graph(supercell=_supercell_for(tiling))
        assert nx.is_connected(graph)
