"""Regression tests for matrix (sheared) supercell support in enumeration (audit L1).

Before the fix, ``_tiling_to_atoms`` unpacked ``na, nb = supercell`` and crashed
with an opaque numpy error on a 2x2 matrix supercell, even though
``Tiling.graph`` accepts matrices. These tests pin that the enumeration/export
atom builder now accepts matrix supercells with graph-consistent atom ordering,
and that the orderly count matches the independent Burnside count.
"""

from __future__ import annotations

import numpy as np

from archimono.assignment.enumeration._helpers import _tiling_to_atoms
from archimono.assignment.enumeration.burnside import BurnsideCounter
from archimono.assignment.enumeration.orderly import OrderlyEnumerator
from archimono.tilings import registry

# snub square 3².4.3.4 with a sheared cell: det = 2, n = 8 atoms.
_KEY = "snub_square"
_MATRIX = np.array([[1, 1], [0, 2]], dtype=np.intp)


def test_atom_order_matches_graph_nodes_for_matrix_supercell() -> None:
    """Atom i sits at the position of graph node i for a matrix supercell."""
    tiling = registry.get(_KEY)
    graph = tiling.graph(_MATRIX)
    atoms = _tiling_to_atoms(tiling, _MATRIX)
    assert len(atoms) == graph.number_of_nodes()
    positions = atoms.get_positions()
    for i in range(graph.number_of_nodes()):
        np.testing.assert_allclose(positions[i][:2], graph.nodes[i]["pos"])


def test_orderly_matrix_supercell_matches_burnside() -> None:
    """Orderly enumeration on a matrix supercell equals the Burnside orbit count."""
    tiling = registry.get(_KEY)
    enumerator = OrderlyEnumerator()
    group = enumerator.get_symmetry_group(tiling, _MATRIX)
    n = len(group[0])
    n_b = n // 2
    orderly = enumerator.enumerate(tiling, n_b=n_b, supercell=_MATRIX)
    burnside = BurnsideCounter.count(n, n_b, group)
    assert len(orderly) == burnside
    # Every representative honours the requested composition.
    for config in orderly:
        assert sum(config) == n_b
