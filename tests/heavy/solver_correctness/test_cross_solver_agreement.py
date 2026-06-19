"""Heavy tests: all solvers agree across all frustrated tilings.

For each frustrated tiling at its minimum valid supercell, exact solvers
must agree. Heuristics must reach near-optimal.
"""

from __future__ import annotations

import pytest

from archimono.assignment import (
    AnnealingSolver,
    BruteforceSolver,
    GreedySolver,
    HadlockSolver,
)
from archimono.assignment.solvers.frontier_dp import FrontierExactSolver
from archimono.tilings import registry

_CASES = [
    ("kagome", (2, 2), 6),
    ("triangular", (2, 2), 2),
    ("truncated_hexagonal", (1, 1), 3),
    ("rhombitrihexagonal", (1, 1), 3),
    ("snub_square", (1, 1), 2),
    ("elongated_triangular", (2, 1), 4),
    ("snub_hexagonal", (1, 1), 3),
]

# Non-balanced compositions n_b = n//3 and 2n//3 for the frustrated tilings,
# complementing the balanced n_b above. See #53.
_NONBALANCED_CASES = [
    ("kagome", (2, 2), 4),  # n=12
    ("kagome", (2, 2), 8),
    ("truncated_hexagonal", (1, 1), 2),  # n=6
    ("truncated_hexagonal", (1, 1), 4),
    ("rhombitrihexagonal", (1, 1), 2),  # n=6
    ("rhombitrihexagonal", (1, 1), 4),
    ("elongated_triangular", (2, 1), 2),  # n=8
    ("elongated_triangular", (2, 1), 5),
    ("snub_hexagonal", (1, 1), 2),  # n=6
    ("snub_hexagonal", (1, 1), 4),
]


class TestExactSolversAgree:
    """BruteforceSolver and FrontierExactSolver agree on all cases."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _ in _CASES],
    )
    def test_bruteforce_vs_frontier(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """Two exact solvers produce the same cut value and respect n_b."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)

        brute_result = BruteforceSolver().solve(graph, ["A", "B"], n_b=n_b)
        frontier_result = FrontierExactSolver().solve(
            graph, ["A", "B"], n_b=n_b
        )

        assert brute_result.cut_value == pytest.approx(
            frontier_result.cut_value
        ), (
            f"{key}: Bruteforce={brute_result.cut_value} vs "
            f"Frontier={frontier_result.cut_value}"
        )
        assert sum(brute_result.labels) == n_b
        assert sum(frontier_result.labels) == n_b


class TestHadlockVsConstrained:
    """Unconstrained Hadlock cut >= constrained brute-force cut on pbc=False."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _ in _CASES],
    )
    def test_hadlock_ge_constrained(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """Unconstrained Hadlock cut >= fixed-composition brute-force cut."""
        tiling = registry.get(key)
        g = tiling.graph(supercell=supercell, pbc=False)
        hadlock = HadlockSolver().solve(g, ["A", "B"])
        brute = BruteforceSolver().solve(g, ["A", "B"], n_b=n_b)
        assert hadlock.cut_value >= brute.cut_value, (
            f"{key}: Hadlock={hadlock.cut_value} < "
            f"Bruteforce={brute.cut_value}"
        )


class TestHeuristicsReachOptimum:
    """Heuristics with sufficient restarts reach near-optimal."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _ in _CASES],
    )
    def test_annealing_reaches_optimum(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """AnnealingSolver with n_restarts=10 reaches >= 95% of exact."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)

        exact_result = FrontierExactSolver().solve(
            graph, ["A", "B"], n_b=n_b
        )
        annealing_result = AnnealingSolver(seed=42, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )

        assert annealing_result.cut_value >= exact_result.cut_value * 0.95, (
            f"{key}: annealing={annealing_result.cut_value}, "
            f"exact={exact_result.cut_value}"
        )
        assert sum(annealing_result.labels) == n_b

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _CASES,
        ids=[f"{k}_{s[0]}x{s[1]}" for k, s, _ in _CASES],
    )
    def test_greedy_reaches_optimum(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """GreedySolver with n_restarts=10 reaches >= 90% of exact."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)

        exact_result = FrontierExactSolver().solve(
            graph, ["A", "B"], n_b=n_b
        )
        greedy_result = GreedySolver(seed=42, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )

        assert greedy_result.cut_value >= exact_result.cut_value * 0.9, (
            f"{key}: greedy={greedy_result.cut_value}, "
            f"exact={exact_result.cut_value}"
        )
        assert sum(greedy_result.labels) == n_b


class TestNonBalancedCompositions:
    """Exact solvers agree and heuristics stay near-optimal at non-balanced n_b."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _NONBALANCED_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}_nb{nb}" for k, s, nb in _NONBALANCED_CASES],
    )
    def test_bruteforce_vs_frontier(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """Two exact solvers agree on the cut value at non-balanced n_b."""
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

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _NONBALANCED_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}_nb{nb}" for k, s, nb in _NONBALANCED_CASES],
    )
    def test_heuristics_reach_optimum(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """Annealing and greedy stay near the exact optimum at non-balanced n_b."""
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)

        exact = FrontierExactSolver().solve(graph, ["A", "B"], n_b=n_b)
        annealing = AnnealingSolver(seed=42, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )
        greedy = GreedySolver(seed=42, n_restarts=10).solve(
            graph, ["A", "B"], n_b=n_b
        )

        assert annealing.cut_value >= exact.cut_value * 0.95, (
            f"{key} n_b={n_b}: annealing={annealing.cut_value}, "
            f"exact={exact.cut_value}"
        )
        assert greedy.cut_value >= exact.cut_value * 0.9, (
            f"{key} n_b={n_b}: greedy={greedy.cut_value}, "
            f"exact={exact.cut_value}"
        )
        assert sum(annealing.labels) == n_b
        assert sum(greedy.labels) == n_b
