"""Contract tests for heuristic quality on representative tilings.

Seeded runs on 3-4 tilings assert bounds against known optima.
"""

from __future__ import annotations

import pytest

from archimono.assignment import AnnealingSolver, GreedySolver
from archimono.tilings import registry

# Representative frustrated tilings with known exact optima.
# (tiling_key, supercell, n_b, known_optimal_cut)
_REFERENCE_CASES = [
    ("kagome", (2, 2), 6, 16),
    ("triangular", (2, 2), 2, 4),
    ("truncated_hexagonal", (1, 1), 3, 6),
]


class TestAnnealingReferenceCases:
    """AnnealingSolver with seed=42 meets bounds on reference cases."""

    @pytest.mark.parametrize(
        "key,supercell,n_b,optimal",
        _REFERENCE_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _, _ in _REFERENCE_CASES],
    )
    def test_seeded_meets_lower_bound(
        self,
        key: str,
        supercell: tuple[int, int],
        n_b: int,
        optimal: int,
    ) -> None:
        """Annealing with seed=42 achieves at least 90% of optimal."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)
        result = AnnealingSolver(seed=42, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )
        assert result.cut_value >= 0.9 * optimal, (
            f"{key}: annealing got {result.cut_value}, "
            f"need >= {0.9 * optimal}"
        )
        assert result.cut_value <= graph.number_of_edges()


class TestGreedyReferenceCases:
    """GreedySolver with seed=42 meets bounds on reference cases."""

    @pytest.mark.parametrize(
        "key,supercell,n_b,optimal",
        _REFERENCE_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _, _ in _REFERENCE_CASES],
    )
    def test_seeded_meets_lower_bound(
        self,
        key: str,
        supercell: tuple[int, int],
        n_b: int,
        optimal: int,
    ) -> None:
        """Greedy with seed=42 achieves at least 80% of optimal."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)
        result = GreedySolver(seed=42, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )
        assert result.cut_value >= 0.8 * optimal, (
            f"{key}: greedy got {result.cut_value}, "
            f"need >= {0.8 * optimal}"
        )
        assert result.cut_value <= graph.number_of_edges()


class TestHeuristicCompositionConstraint:
    """Heuristics respect explicit n_b constraint."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        [("kagome", (2, 2), 6), ("kagome", (2, 2), 4), ("kagome", (2, 2), 3)],
        ids=["balanced", "nb4", "nb3"],
    )
    def test_annealing_respects_nb(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """AnnealingSolver result has exactly n_b label-1 entries."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)
        result = AnnealingSolver(seed=42, n_restarts=3).solve(
            graph, ["A", "B"], n_b=n_b
        )
        assert sum(result.labels) == n_b

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        [("kagome", (2, 2), 6), ("kagome", (2, 2), 4), ("kagome", (2, 2), 3)],
        ids=["balanced", "nb4", "nb3"],
    )
    def test_greedy_respects_nb(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """GreedySolver result has exactly n_b label-1 entries."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)
        result = GreedySolver(seed=42, n_restarts=3).solve(
            graph, ["A", "B"], n_b=n_b
        )
        assert sum(result.labels) == n_b
