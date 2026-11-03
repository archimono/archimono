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

    def test_kagome_2x2_seeded(self) -> None:
        """Annealing on kagome 2x2 (n=12, n_b=1) with seed=42.

        Bound: with n_b=1, exactly one node is labeled B and every other
        node is A, so every one of that node's incident edges is cut and no
        other edge can be -- the optimal (and only achievable) cut value is
        always exactly the coordination number z=4, regardless of which
        node is chosen (kagome is vertex-transitive).
        """
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        r1 = AnnealingSolver(seed=42, n_restarts=5).solve(
            graph, ["A", "B"], n_b=1
        )
        r2 = AnnealingSolver(seed=42, n_restarts=5).solve(
            graph, ["A", "B"], n_b=1
        )
        assert r1.cut_value == pytest.approx(4.0)
        assert r1.cut_value <= graph.number_of_edges()
        assert r1.labels == r2.labels


class TestGreedySeeded:
    """GreedySolver produces deterministic results with fixed seed."""

    def test_kagome_2x2_seeded(self) -> None:
        """Greedy on kagome 2x2 (n=12, n_b=1) with seed=42.

        Same bound as annealing: optimal = z = 4.
        """
        tiling = registry.get("kagome")
        graph = tiling.graph(supercell=(2, 2))
        r1 = GreedySolver(seed=42, n_restarts=5).solve(
            graph, ["A", "B"], n_b=1
        )
        r2 = GreedySolver(seed=42, n_restarts=5).solve(
            graph, ["A", "B"], n_b=1
        )
        assert r1.cut_value == pytest.approx(4.0)
        assert r1.cut_value <= graph.number_of_edges()
        assert r1.labels == r2.labels
