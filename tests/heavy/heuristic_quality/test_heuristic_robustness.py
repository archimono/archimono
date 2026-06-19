"""Heavy heuristic tests: all frustrated tilings, multi-seed, bounds vs exact.

Tests that heuristics reliably find optimal solutions across all frustrated
tilings with multiple seeds.
"""

from __future__ import annotations

import pytest

from archimono.assignment import AnnealingSolver, GreedySolver
from archimono.assignment.solvers.frontier_dp import FrontierExactSolver
from archimono.tilings import registry

# (tiling_key, supercell, n_b)
_FRUSTRATED_CASES = [
    ("triangular", (2, 2), 2),
    ("kagome", (1, 1), 1),
    ("elongated_triangular", (2, 1), 4),
    ("snub_square", (1, 1), 2),
    ("truncated_hexagonal", (1, 1), 3),
    ("rhombitrihexagonal", (1, 1), 3),
    ("snub_hexagonal", (1, 1), 3),
]

_SEEDS = [0, 1, 2, 7, 42]


class TestAnnealingMultiSeed:
    """AnnealingSolver reaches optimum across multiple seeds."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _FRUSTRATED_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _ in _FRUSTRATED_CASES],
    )
    def test_annealing_reaches_optimum(
        self, key: str, supercell: tuple[int, int], n_b: int, seed: int
    ) -> None:
        """Annealing with 10 restarts finds exact optimum on small case."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)

        exact = FrontierExactSolver().solve(graph, ["A", "B"], n_b=n_b)
        result = AnnealingSolver(seed=seed, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )

        assert result.cut_value == exact.cut_value, (
            f"{key} seed={seed}: got {result.cut_value}, "
            f"expected {exact.cut_value}"
        )


class TestGreedyMultiSeed:
    """GreedySolver reaches optimum across multiple seeds."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _FRUSTRATED_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _ in _FRUSTRATED_CASES],
    )
    def test_greedy_reaches_optimum(
        self, key: str, supercell: tuple[int, int], n_b: int, seed: int
    ) -> None:
        """Greedy with 10 restarts finds exact optimum on small case."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)

        exact = FrontierExactSolver().solve(graph, ["A", "B"], n_b=n_b)
        result = GreedySolver(seed=seed, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )

        assert result.cut_value == exact.cut_value, (
            f"{key} seed={seed}: got {result.cut_value}, "
            f"expected {exact.cut_value}"
        )


class TestHeuristicUpperBound:
    """Heuristic results never exceed total edge count."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _FRUSTRATED_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _ in _FRUSTRATED_CASES],
    )
    def test_cut_value_bounded(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """Cut value <= |E| for both heuristics."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)
        total_edges = graph.number_of_edges()

        ann = AnnealingSolver(seed=42, n_restarts=3).solve(
            graph, ["A", "B"], n_b=n_b
        )
        grd = GreedySolver(seed=42, n_restarts=3).solve(
            graph, ["A", "B"], n_b=n_b
        )

        assert ann.cut_value <= total_edges
        assert grd.cut_value <= total_edges
