"""Fast smoke tests for tiling graph construction.

Verifies that graph() returns correct types and basic properties
on 1-2 tilings without heavy computation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from archimono.tilings import Tiling, registry


class TestGraphConstruction:
    """Tiling.graph() returns correct type and basic structure."""

    def test_returns_networkx_graph(self) -> None:
        """graph() returns a networkx Graph instance."""
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        assert isinstance(graph, nx.Graph)

    def test_node_count_matches_formula(self) -> None:
        """Node count = n_vertices * na * nb for a simple supercell."""
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        assert graph.number_of_nodes() == tiling.n_vertices * 2 * 2

    def test_edge_count_positive(self) -> None:
        """Graph has at least one edge."""
        tiling = registry.get("hexagonal")
        graph = tiling.graph(supercell=(2, 2))
        assert graph.number_of_edges() > 0

    def test_no_self_loops(self) -> None:
        """Graph has no self-loops."""
        tiling = registry.get("triangular")
        graph = tiling.graph(supercell=(2, 2))
        assert nx.number_of_selfloops(graph) == 0

    def test_pbc_false_fewer_edges(self) -> None:
        """pbc=False produces fewer edges than pbc=True."""
        tiling = registry.get("kagome")
        g_pbc = tiling.graph(supercell=(2, 2))
        g_open = tiling.graph(supercell=(2, 2), pbc=False)
        assert g_open.number_of_nodes() == g_pbc.number_of_nodes()
        assert g_open.number_of_edges() < g_pbc.number_of_edges()

    def test_pbc_metadata(self) -> None:
        """graph() stores the pbc flag in graph metadata."""
        tiling = registry.get("kagome")
        g_true = tiling.graph(supercell=(2, 2))
        g_false = tiling.graph(supercell=(2, 2), pbc=False)
        assert g_true.graph["pbc"] is True
        assert g_false.graph["pbc"] is False

    def test_pbc_false_is_planar(self) -> None:
        """pbc=False graph from a planar tiling is planar."""
        tiling = registry.get("kagome")
        g = tiling.graph(supercell=(2, 2), pbc=False)
        assert nx.check_planarity(g)[0]

    def test_lattice_vectors_are_2d(self) -> None:
        """Lattice vectors are 2D numpy arrays."""
        tiling = registry.get("kagome")
        vectors = tiling.lattice_vectors
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (2, 2)


class TestRegistryBasics:
    """Registry provides consistent tiling access."""

    def test_available_returns_sorted_list(self) -> None:
        """available() returns a sorted list of tiling keys."""
        keys = registry.available()
        assert keys == sorted(keys)
        assert len(keys) >= 11

    def test_get_returns_correct_type(self) -> None:
        """get() returns a Tiling instance."""
        t = registry.get("kagome")
        assert isinstance(t, Tiling)

    def test_invalid_key_raises(self) -> None:
        """Unknown tiling key raises KeyError."""
        with pytest.raises(KeyError):
            registry.get("nonexistent_tiling")
