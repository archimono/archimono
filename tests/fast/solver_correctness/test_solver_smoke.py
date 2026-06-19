"""Fast smoke tests for all solvers.

Each solver is tested on a trivial input (C4 cycle) to verify it returns
a valid AssignmentResult with non-negative cut value.
"""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.assignment import (
    AnnealingSolver,
    BipartiteAssigner,
    BruteforceSolver,
    GreedySolver,
    HadlockSolver,
)
from archimono.assignment.solvers.frontier_dp import FrontierExactSolver
from archimono.tilings import registry


class TestSolverReturnsValidResult:
    """Every solver returns a structurally valid AssignmentResult."""

    def test_hadlock_on_c4(self, c4_graph: nx.Graph[int]) -> None:
        """HadlockSolver produces valid result on C4."""
        result = HadlockSolver().solve(c4_graph, ["A", "B"])
        assert result.cut_value >= 0
        assert len(result.labels) == 4
        assert all(lbl in (0, 1) for lbl in result.labels)

    def test_bruteforce_on_c4(self, c4_graph: nx.Graph[int]) -> None:
        """BruteforceSolver produces valid result on C4, defaults n_b to n//2."""
        result = BruteforceSolver().solve(c4_graph, ["A", "B"])
        assert result.cut_value >= 0
        assert len(result.labels) == 4
        assert all(lbl in (0, 1) for lbl in result.labels)
        assert sum(result.labels) == 2

    def test_annealing_on_c4(self, c4_graph: nx.Graph[int]) -> None:
        """AnnealingSolver produces valid result on C4."""
        result = AnnealingSolver(seed=42, n_restarts=1).solve(c4_graph, ["A", "B"])
        assert result.cut_value >= 0
        assert len(result.labels) == 4
        assert all(lbl in (0, 1) for lbl in result.labels)

    def test_greedy_on_c4(self, c4_graph: nx.Graph[int]) -> None:
        """GreedySolver produces valid result on C4."""
        result = GreedySolver(seed=42, n_restarts=1).solve(c4_graph, ["A", "B"])
        assert result.cut_value >= 0
        assert len(result.labels) == 4
        assert all(lbl in (0, 1) for lbl in result.labels)

    def test_bipartite_on_hexagonal(self) -> None:
        """BipartiteAssigner produces valid result on hexagonal 2x2."""
        tiling = registry.get("hexagonal")
        result = BipartiteAssigner.assign(tiling, n_b=4, supercell=(2, 2))
        assert result.cut_value >= 0
        assert len(result.labels) == 8
        assert all(lbl in (0, 1) for lbl in result.labels)

    def test_frontier_dp_on_c4(self, c4_graph: nx.Graph[int]) -> None:
        """FrontierExactSolver produces valid result on C4."""
        result = FrontierExactSolver().solve(c4_graph, ["A", "B"])
        assert result.cut_value >= 0
        assert len(result.labels) == 4
        assert all(lbl in (0, 1) for lbl in result.labels)


class TestSolverOptimalityOnC4:
    """Exact and bipartite solvers find provably optimal cuts."""

    def test_hadlock_optimal(self, c4_graph: nx.Graph[int]) -> None:
        """HadlockSolver finds optimal cut on C4."""
        result = HadlockSolver().solve(c4_graph, ["A", "B"])
        assert result.cut_value == pytest.approx(4.0)

    def test_bipartite_optimal(self) -> None:
        """BipartiteAssigner achieves full cut on hexagonal 2x2."""
        tiling = registry.get("hexagonal")
        result = BipartiteAssigner.assign(tiling, n_b=4, supercell=(2, 2))
        assert result.cut_value == pytest.approx(12.0)
        assert result.n_frustrated == 0
