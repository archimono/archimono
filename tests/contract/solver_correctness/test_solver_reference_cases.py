"""Contract tests for solver correctness on reference cases.

Exact solvers match known optima. Bipartite solver is provably optimal.
Result labels form a valid partition.
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


class TestExactSolverReferenceValues:
    """Exact solvers match known MAX-CUT optima."""

    def test_kagome_2x2_bruteforce(self) -> None:
        """BruteforceSolver on kagome 2x2: k*=16 (verified by brute-force)."""
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        result = BruteforceSolver().solve(graph, ["A", "B"], n_b=6)
        assert result.cut_value == pytest.approx(16.0)
        assert result.n_frustrated == 8

    def test_hadlock_on_kagome_pbc_false(self) -> None:
        """HadlockSolver unconstrained cut >= constrained on kagome pbc=False."""
        tiling = registry.get("kagome")
        g = tiling.graph(supercell=(2, 2), pbc=False)
        hadlock = HadlockSolver().solve(g, ["A", "B"])
        brute = BruteforceSolver().solve(
            g, ["A", "B"], n_b=g.number_of_nodes() // 2
        )
        assert hadlock.cut_value >= brute.cut_value
        assert len(hadlock.labels) == g.number_of_nodes()
        assert all(lbl in (0, 1) for lbl in hadlock.labels)

    def test_bruteforce_on_non_planar_pbc_graph(self) -> None:
        """BruteforceSolver works on non-planar PBC graph (key design property)."""
        tiling = registry.get("square")
        graph = tiling.graph(supercell=(2, 2))
        assert not nx.check_planarity(graph)[0]
        result = BruteforceSolver().solve(graph, ["A", "B"], n_b=4)
        assert result.cut_value == pytest.approx(16.0)

    def test_kagome_2x2_frontier_dp(self) -> None:
        """FrontierExactSolver agrees with BruteforceSolver on kagome 2x2."""
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        result = FrontierExactSolver().solve(graph, ["A", "B"], n_b=6)
        assert result.cut_value == pytest.approx(16.0)
        assert sum(result.labels) == 6


class TestBipartiteOptimality:
    """BipartiteAssigner is provably optimal on all bipartite tilings."""

    @pytest.mark.parametrize(
        "key,edges_2x2",
        [
            ("hexagonal", 12),
            ("square", 16),
            ("truncated_square", 24),
            ("truncated_trihexagonal", 72),
        ],
        ids=["hexagonal", "square", "truncated_square", "truncated_trihex"],
    )
    def test_full_cut_on_bipartite_tiling(
        self, key: str, edges_2x2: int
    ) -> None:
        """Bipartite tiling at 2x2: cut == |E|, n_frustrated == 0."""
        tiling = registry.get(key)
        n = tiling.n_vertices * 4
        result = BipartiteAssigner.assign(tiling, n_b=n // 2, supercell=(2, 2))
        assert result.cut_value == pytest.approx(edges_2x2)
        assert result.n_frustrated == 0
        assert sum(result.labels) == n // 2


class TestResultLabelsValid:
    """Solver results contain valid binary partitions."""

    @pytest.mark.parametrize("key", ["kagome", "triangular"])
    def test_labels_are_binary(self, key: str) -> None:
        """Labels contain only 0 and 1."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=(2, 2))
        n_b = graph.number_of_nodes() // 2
        result = BruteforceSolver().solve(graph, ["A", "B"], n_b=n_b)
        assert all(lbl in (0, 1) for lbl in result.labels)

    @pytest.mark.parametrize("key", ["kagome", "triangular"])
    def test_labels_length_matches_nodes(self, key: str) -> None:
        """Labels tuple has one entry per node."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=(2, 2))
        n_b = graph.number_of_nodes() // 2
        result = BruteforceSolver().solve(graph, ["A", "B"], n_b=n_b)
        assert len(result.labels) == graph.number_of_nodes()

    @pytest.mark.parametrize("key", ["kagome", "triangular"])
    def test_nb_constraint_respected(self, key: str) -> None:
        """When n_b is specified, exactly n_b nodes have label 1."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=(2, 2))
        n_b = graph.number_of_nodes() // 2
        result = BruteforceSolver().solve(graph, ["A", "B"], n_b=n_b)
        assert sum(result.labels) == n_b


# Non-balanced compositions (n_b != n/2) are otherwise undertested; see #53.
class TestNonBalancedExactSolvers:
    """Exact solvers agree and respect composition at non-balanced n_b."""

    # (key, supercell, n_b): kagome 2x2 has n=12 (n//3=4, 2n//3=8).
    _CASES = [
        ("kagome", (2, 2), 4),
        ("kagome", (2, 2), 8),
        ("triangular", (2, 2), 1),
        ("triangular", (2, 2), 3),
    ]

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _CASES,
        ids=[f"{k}_{s[0]}x{s[1]}_nb{nb}" for k, s, nb in _CASES],
    )
    def test_bruteforce_vs_frontier_nonbalanced(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """Bruteforce and frontier DP agree, and exactly n_b sites are B."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)

        brute = BruteforceSolver().solve(graph, ["A", "B"], n_b=n_b)
        frontier = FrontierExactSolver().solve(graph, ["A", "B"], n_b=n_b)

        assert brute.cut_value == pytest.approx(frontier.cut_value), (
            f"{key} n_b={n_b}: brute={brute.cut_value} "
            f"!= frontier={frontier.cut_value}"
        )
        assert sum(brute.labels) == n_b
        assert sum(frontier.labels) == n_b


class TestSingleMinoritySpecies:
    """Edge compositions n_b=1 and n_b=n-1: every solver reaches the optimum.

    Covered for triangular 2x2 (n=4) per #53's edge-coverage request. The
    optimum is whatever the exact bruteforce solver finds; both the other exact
    solver and the heuristics must match it (the search space is tiny here).
    """

    @pytest.mark.parametrize("n_b", [1, 3], ids=["nb1", "nb_n_minus_1"])
    def test_triangular_single_species(self, n_b: int) -> None:
        """All four solvers agree on the optimal cut and respect composition."""
        tiling = registry.get("triangular")
        graph = tiling.graph(supercell=(2, 2))

        brute = BruteforceSolver().solve(graph, ["A", "B"], n_b=n_b)
        frontier = FrontierExactSolver().solve(graph, ["A", "B"], n_b=n_b)
        greedy = GreedySolver(seed=42, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )
        annealing = AnnealingSolver(seed=42, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )

        optimum = brute.cut_value
        assert frontier.cut_value == pytest.approx(optimum)
        assert greedy.cut_value == pytest.approx(optimum)
        assert annealing.cut_value == pytest.approx(optimum)
        for result in (brute, frontier, greedy, annealing):
            assert sum(result.labels) == n_b


class TestBipartiteRejectsNonBalanced:
    """BipartiteAssigner requires n_b = n/2 and rejects other compositions."""

    @pytest.mark.parametrize("key", ["square", "hexagonal"])
    def test_nonbalanced_nb_raises(self, key: str) -> None:
        """A non-balanced n_b on a bipartite tiling raises ValueError."""
        tiling = registry.get(key)
        n = tiling.n_vertices * 4  # 2x2 supercell
        with pytest.raises(ValueError, match="not balanced"):
            BipartiteAssigner.assign(tiling, n_b=n // 2 - 1, supercell=(2, 2))
