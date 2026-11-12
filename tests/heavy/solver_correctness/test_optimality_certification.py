"""Heavy tests: brute-force confirms exact solver optimality.

On small enough cases, enumerate all C(n, n_b) assignments and verify
the exact solver found the true maximum.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import pytest

from archimono.assignment import BruteforceSolver
from archimono.tilings import registry

# Cases small enough for brute-force certification (C(n, n_b) <= 1000).
# kagome, triangular, and snub_square alias at their previous sizes here
# (edge instances collapse under PBC -- see Tiling._graph_matrix); bumped to
# their smallest supported supercells. n_b is kept as-is (not rebalanced to
# n//2) so C(n, n_b) stays within this file's budget.
_CERTIFIABLE_CASES = [
    ("kagome", (2, 2), 1),
    ("triangular", (3, 3), 2),
    ("snub_square", (2, 2), 2),
    ("truncated_hexagonal", (1, 1), 3),
    ("rhombitrihexagonal", (1, 1), 3),
    ("kagome", (2, 2), 6),
]


def _brute_force_max_cut(graph: nx.Graph[int], n_b: int) -> int:
    """Enumerate all C(n, n_b) assignments, return maximum cut value."""
    edges = list(graph.edges())
    best = 0
    for b_nodes in combinations(range(graph.number_of_nodes()), n_b):
        b_set = set(b_nodes)
        cut = sum(1 for u, v in edges if (u in b_set) != (v in b_set))
        best = max(best, cut)
    return best


class TestBruteForceConfirmsSolver:
    """Brute-force enumeration confirms BruteforceSolver is optimal."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _CERTIFIABLE_CASES,
        ids=[
            f"{k}_{s[0]}x{s[1]}_nb{nb}" for k, s, nb in _CERTIFIABLE_CASES
        ],
    )
    def test_solver_matches_brute_force(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        tiling = registry.get(key)
        graph = tiling.graph(supercell=supercell)

        solver_result = BruteforceSolver().solve(graph, ["A", "B"], n_b=n_b)
        brute_force_opt = _brute_force_max_cut(graph, n_b)

        assert int(solver_result.cut_value) == brute_force_opt, (
            f"{key} {supercell} n_b={n_b}: "
            f"solver={solver_result.cut_value}, brute_force={brute_force_opt}"
        )
