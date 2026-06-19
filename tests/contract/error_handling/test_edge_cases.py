"""Contract tests for edge-case error handling.

Boundary conditions that are valid-but-extreme or just-outside-valid.
"""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.assignment import (
    AnnealingSolver,
    BipartiteAssigner,
    BruteforceSolver,
    HadlockSolver,
)
from archimono.tilings import registry


class TestNbBoundaryConditions:
    """n_b at boundaries: n_b=1 (minimum valid) and n_b=n-1 (maximum valid)."""

    def test_bruteforce_nb_equals_1(self) -> None:
        """n_b=1 is valid (one node in partition B)."""
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        result = BruteforceSolver().solve(graph, ["A", "B"], n_b=1)
        assert sum(result.labels) == 1

    def test_bruteforce_nb_equals_n_minus_1(self) -> None:
        """n_b=n-1 is valid (one node in partition A)."""
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        n = graph.number_of_nodes()
        result = BruteforceSolver().solve(graph, ["A", "B"], n_b=n - 1)
        assert sum(result.labels) == n - 1

    def test_annealing_nb_equals_1(self) -> None:
        """Heuristic solver handles extreme n_b=1."""
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        result = AnnealingSolver(seed=42).solve(graph, ["A", "B"], n_b=1)
        assert sum(result.labels) == 1

    def test_annealing_nb_equals_n_minus_1(self) -> None:
        """Heuristic solver handles extreme n_b=n-1."""
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        n = graph.number_of_nodes()
        result = AnnealingSolver(seed=42).solve(
            graph, ["A", "B"], n_b=n - 1
        )
        assert sum(result.labels) == n - 1


class TestNonPlanarRejection:
    """HadlockSolver rejects non-planar graphs from tiling PBC."""

    @pytest.mark.parametrize(
        "key,supercell",
        [
            ("square", (2, 2)),
            ("elongated_triangular", (2, 1)),
            ("snub_hexagonal", (1, 1)),
        ],
        ids=["square_2x2", "elongated_tri_2x1", "snub_hex_1x1"],
    )
    def test_pbc_non_planar_rejected(
        self, key: str, supercell: tuple[int, int]
    ) -> None:
        """PBC wrapping can make a planar tiling non-planar; solver rejects."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)
        assert not nx.check_planarity(graph)[0], (
            f"{key} at {supercell} expected non-planar under PBC"
        )
        with pytest.raises(ValueError, match="planar"):
            HadlockSolver().solve(graph, ["A", "B"])


class TestInvalidSupercellDimensions:
    """Tiling.graph() rejects invalid supercell dimensions."""

    @pytest.mark.parametrize(
        "supercell",
        [(0, 2), (2, 0), (-1, 2), (2, -1)],
        ids=["zero_na", "zero_nb", "neg_na", "neg_nb"],
    )
    def test_rejects_non_positive_supercell(
        self, supercell: tuple[int, int]
    ) -> None:
        """Supercell dimensions must be positive integers."""
        tiling = registry.get("kagome")
        with pytest.raises(ValueError, match="positive"):
            tiling.graph(supercell=supercell)


class TestBipartiteAssignerOnFrustrated:
    """BipartiteAssigner rejects frustrated (non-bipartite) tilings."""

    def test_rejects_kagome(self) -> None:
        """Kagome is not bipartite; BipartiteAssigner raises."""
        tiling = registry.get("kagome")
        with pytest.raises(ValueError, match="non-bipartite"):
            BipartiteAssigner.assign(tiling, n_b=6, supercell=(2, 2))
