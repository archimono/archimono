"""Each tiling's edge list is exactly its set of unit-distance vertex pairs.

Symmetry permutations come from spglib, which reads vertex positions only. A
symmetry of the positions is an isometry, so it maps edges to edges exactly
when the edge list is the set of vertex pairs at the edge length. This checks
that the hand-written edge list and the coordinates agree.
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from archimono.tilings import registry

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
_EDGE_LENGTH = 1.0  # registry.get default
_REACH = 2  # neighbour cells searched in each direction
_TOL = 1e-6

Edge = tuple[int, int, int, int]


def _canonical(edge: Edge) -> Edge:
    i, j, da, db = edge
    return min((i, j, da, db), (j, i, -da, -db))


@pytest.mark.parametrize("key", _TILINGS)
def test_edge_list_equals_unit_distance_pairs(key: str) -> None:
    tiling = registry.get(key)
    a, b = tiling.lattice_vectors[0], tiling.lattice_vectors[1]

    def cart(v: int, da: int, db: int) -> npt.NDArray[np.float64]:
        frac = tiling.vertices[v]
        return np.asarray((frac[0] + da) * a + (frac[1] + db) * b, dtype=float)

    listed = [_canonical(e) for e in tiling.edges]
    assert len(set(listed)) == len(listed), "duplicate edge"
    assert len(listed) == tiling.n_vertices * tiling.coordination // 2
    assert all(abs(da) <= _REACH and abs(db) <= _REACH for _, _, da, db in listed)

    unit_pairs: set[Edge] = set()
    offsets = range(-_REACH, _REACH + 1)
    for i, j in itertools.product(range(tiling.n_vertices), repeat=2):
        for da, db in itertools.product(offsets, repeat=2):
            if (i, da, db) == (j, 0, 0):
                continue
            d = float(np.linalg.norm(cart(j, da, db) - cart(i, 0, 0)))
            assert d > _EDGE_LENGTH - _TOL, (
                f"vertices closer than an edge: {(i, j, da, db)}"
            )
            if d < _EDGE_LENGTH + _TOL:
                unit_pairs.add(_canonical((i, j, da, db)))

    assert set(listed) == unit_pairs
