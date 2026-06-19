"""Regression test: diagonal-tuple and diagonal-matrix supercells agree (audit L2).

Before the fix, the diagonal path computed positions as
``(frac + ia) / na * na * a_vec`` (a floating-point no-op that could differ from
the clean form in the last bit). After simplifying it to match the matrix path,
the tuple ``(na, nb)`` and the equivalent diagonal matrix produce bit-identical
positions.
"""

from __future__ import annotations

import numpy as np

from archimono.tilings import registry


def test_diagonal_tuple_and_matrix_positions_identical() -> None:
    """``graph((2, 2))`` and ``graph([[2, 0], [0, 2]])`` give identical positions."""
    tiling = registry.get("kagome")
    g_tuple = tiling.graph((2, 2))
    g_matrix = tiling.graph(np.array([[2, 0], [0, 2]], dtype=np.intp))

    assert set(g_tuple.nodes) == set(g_matrix.nodes)
    for node in g_tuple.nodes:
        np.testing.assert_array_equal(
            g_tuple.nodes[node]["pos"], g_matrix.nodes[node]["pos"]
        )
