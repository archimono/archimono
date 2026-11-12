"""Symmetry permutations used for orbit reduction are graph automorphisms.

Orbit reduction is valid only if each permutation ``g`` from
``get_symmetry_group`` maps every edge ``(u, v)`` to an edge
``(g(u), g(v))``. The permutations index spglib sites and the edges index
``Tiling.graph`` nodes, so this also checks that the two orderings agree.
"""

from __future__ import annotations

import pytest

from archimono.assignment import OrderlyEnumerator
from archimono.tilings import registry
from archimono.tilings.base import SupercellLike

# Smallest supported diagonal supercell per tiling, plus one shear matrix.
_CASES: list[tuple[str, SupercellLike]] = [
    ("truncated_hexagonal", (1, 1)),
    ("snub_hexagonal", (1, 1)),
    ("rhombitrihexagonal", (1, 1)),
    ("truncated_trihexagonal", (1, 1)),
    ("elongated_triangular", (3, 1)),
    ("snub_square", (2, 2)),
    ("kagome", (2, 2)),
    ("truncated_square", (2, 2)),
    ("square", (2, 2)),
    ("hexagonal", (2, 2)),
    ("triangular", (3, 3)),
    ("truncated_square", [[2, 1], [0, 3]]),
]


@pytest.mark.parametrize(("key", "supercell"), _CASES)
def test_symmetry_permutations_preserve_edges(
    key: str, supercell: SupercellLike
) -> None:
    tiling = registry.get(key)
    graph = tiling.graph(supercell)
    n = graph.number_of_nodes()
    edges = {frozenset(e) for e in graph.edges()}
    group = OrderlyEnumerator().get_symmetry_group(tiling, supercell)

    assert tuple(range(n)) in group
    assert all(sorted(g) == list(range(n)) for g in group)
    bad = [
        g
        for g in group
        if any(frozenset((g[u], g[v])) not in edges for u, v in graph.edges())
    ]
    assert bad == []
