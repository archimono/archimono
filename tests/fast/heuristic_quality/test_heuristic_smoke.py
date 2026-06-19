"""Fast smoke tests for heuristic solvers with deterministic seeding.

Each heuristic is tested on a single small input with a fixed seed.
Tests assert both the exact seeded value AND a provable bound.
"""

from __future__ import annotations

import pytest

from archimono.assignment import AnnealingSolver, GreedySolver
from archimono.tilings import registry


class TestAnnealingSeeded:
    """AnnealingSolver produces deterministic results with fixed seed."""

    def test_kagome_1x1_seeded(self) -> None:
        """Annealing on kagome 1x1 (n=3, n_b=1) with seed=42.

        Bound: optimal cut on 3-node triangle with 1 vs 2 split = 2 edges.
        (every split cuts exactly 2 of 3 edges in a triangle)
        """
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(1, 1))
        r1 = AnnealingSolver(seed=42, n_restarts=5).solve(
            graph, ["A", "B"], n_b=1
        )
        r2 = AnnealingSolver(seed=42, n_restarts=5).solve(
            graph, ["A", "B"], n_b=1
        )
        assert r1.cut_value == pytest.approx(2.0)
        assert r1.cut_value <= graph.number_of_edges()
        assert r1.labels == r2.labels


class TestGreedySeeded:
    """GreedySolver produces deterministic results with fixed seed."""

    def test_kagome_1x1_seeded(self) -> None:
        """Greedy on kagome 1x1 (n=3, n_b=1) with seed=42.

        Same bound as annealing: optimal = 2.
        """
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(1, 1))
        r1 = GreedySolver(seed=42, n_restarts=5).solve(
            graph, ["A", "B"], n_b=1
        )
        r2 = GreedySolver(seed=42, n_restarts=5).solve(
            graph, ["A", "B"], n_b=1
        )
        assert r1.cut_value == pytest.approx(2.0)
        assert r1.cut_value <= graph.number_of_edges()
        assert r1.labels == r2.labels
