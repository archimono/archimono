"""Symmetry permutations are graph automorphisms for every valid 24-node supercell.

Orbit reduction is valid only if each permutation ``g`` from
``get_symmetry_group`` maps every edge ``(u, v)`` to an edge
``(g(u), g(v))``. This covers every valid diagonal and shear supercell matrix
with 24 nodes, the smallest size every tiling supports.
"""

from __future__ import annotations

import pytest

from archimono.assignment import OrderlyEnumerator
from archimono.tilings import registry
from archimono.tilings.base import valid_supercell_matrices

_TILINGS = (
    "truncated_hexagonal",
    "snub_hexagonal",
    "rhombitrihexagonal",
    "truncated_trihexagonal",
    "elongated_triangular",
    "snub_square",
    "kagome",
    "truncated_square",
    "square",
    "hexagonal",
    "triangular",
)


@pytest.mark.parametrize("key", _TILINGS)
def test_every_valid_supercell_at_24_nodes(key: str) -> None:
    tiling = registry.get(key)
    matrices = valid_supercell_matrices(tiling, 24)
    assert matrices
    for s in matrices:
        graph = tiling.graph(s)
        edges = {frozenset(e) for e in graph.edges()}
        group = OrderlyEnumerator().get_symmetry_group(tiling, s)
        bad = [
            g
            for g in group
            if any(frozenset((g[u], g[v])) not in edges for u, v in graph.edges())
        ]
        assert bad == [], s.tolist()
