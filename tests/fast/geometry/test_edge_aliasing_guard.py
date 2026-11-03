"""Regression tests for the edge-aliasing guard in ``Tiling._graph_matrix``.

For small enough supercells, two periodic edge instances (either two
different unit-cell edge templates, or the same template applied from two
different starting cells) wrap onto the same node pair under PBC.
``networkx.Graph.add_edge`` silently drops the second one, so the graph
ends up with fewer edges than physically exist. ``Tiling.graph()`` now
raises ``ValueError`` for these cases instead (issue #78) -- these tests
pin that guard directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from archimono.tilings import registry
from archimono.tilings.base import _matrix_supercell_is_valid

# (tiling_key, supercell) pairs confirmed to alias under PBC.
_ALIASED_DIAGONAL_CASES = [
    ("triangular", (2, 2)),
    ("elongated_triangular", (2, 1)),
    ("elongated_triangular", (2, 2)),
    ("hexagonal", (1, 1)),
    ("square", (1, 1)),
    ("kagome", (1, 1)),
    ("snub_square", (1, 1)),
]

# (tiling_key, supercell) pairs confirmed clean -- the smallest supported
# supercell for each affected tiling, plus a representative unaffected one.
_CLEAN_CASES = [
    ("triangular", (3, 3)),
    ("elongated_triangular", (3, 1)),
    ("hexagonal", (2, 2)),
    ("square", (2, 2)),
    ("kagome", (2, 2)),
    ("snub_square", (2, 2)),
    ("truncated_hexagonal", (1, 1)),
]


@pytest.mark.parametrize(
    "key,supercell",
    _ALIASED_DIAGONAL_CASES,
    ids=[f"{k}_{s[0]}x{s[1]}" for k, s in _ALIASED_DIAGONAL_CASES],
)
def test_aliased_diagonal_supercell_raises(
    key: str, supercell: tuple[int, int]
) -> None:
    """Confirmed-aliased diagonal supercells raise ValueError under PBC."""
    tiling = registry.get(key)
    with pytest.raises(ValueError, match="alias|multiplicity"):
        tiling.graph(supercell)


def test_aliased_matrix_supercell_raises() -> None:
    """A sheared matrix supercell that aliases also raises ValueError."""
    tiling = registry.get("snub_square")
    with pytest.raises(ValueError, match="alias|multiplicity"):
        tiling.graph(np.array([[1, 1], [0, 2]], dtype=np.intp))


@pytest.mark.parametrize(
    "key,supercell",
    [("triangular", (2, 2)), ("hexagonal", (1, 1)), ("kagome", (1, 1))],
    ids=["triangular_2x2", "hexagonal_1x1", "kagome_1x1"],
)
def test_aliasing_guard_does_not_apply_without_pbc(
    key: str, supercell: tuple[int, int]
) -> None:
    """Aliasing is a PBC-only artefact: pbc=False still builds these cells.

    With pbc=False, boundary-crossing edges are omitted rather than
    wrapped, so two edge instances can never be forced onto the same node
    pair -- there is nothing to alias.
    """
    graph = registry.get(key).graph(supercell, pbc=False)
    assert graph.number_of_nodes() > 0


@pytest.mark.parametrize(
    "key,supercell",
    _CLEAN_CASES,
    ids=[f"{k}_{s[0]}x{s[1]}" for k, s in _CLEAN_CASES],
)
def test_clean_supercell_matches_physical_edge_count(
    key: str, supercell: tuple[int, int]
) -> None:
    """At a supported supercell, edge count exactly matches N * z / 2."""
    tiling = registry.get(key)
    graph = tiling.graph(supercell)
    expected = graph.number_of_nodes() * tiling.coordination // 2
    assert graph.number_of_edges() == expected


def test_matrix_supercell_is_valid_agrees_with_graph_construction() -> None:
    """_matrix_supercell_is_valid predicts graph() success for all three modes."""
    cases = [
        # (tiling_key, matrix, expect_valid, reason)
        ("triangular", [[1, 0], [0, 5]], False, "self-loop"),
        ("truncated_square", [[3, 0], [0, 3]], False, "bipartite parity"),
        ("triangular", [[2, 0], [0, 2]], False, "edge aliasing"),
        ("kagome", [[2, 0], [0, 2]], True, "clean control"),
    ]
    for key, matrix, expect_valid, reason in cases:
        tiling = registry.get(key)
        s_mat = np.array(matrix, dtype=np.intp)

        is_valid = _matrix_supercell_is_valid(tiling, s_mat)
        assert is_valid == expect_valid, (
            f"{key} {matrix} ({reason}): "
            f"_matrix_supercell_is_valid returned {is_valid}, "
            f"expected {expect_valid}"
        )

        try:
            tiling.graph(s_mat)
            graph_succeeds = True
        except ValueError:
            graph_succeeds = False
        assert is_valid == graph_succeeds, (
            f"{key} {matrix} ({reason}): _matrix_supercell_is_valid="
            f"{is_valid} disagrees with actual graph() success="
            f"{graph_succeeds}"
        )


def test_reverse_orientation_collision_is_detected() -> None:
    """A collision generated once as (u, v) and once as (v, u) is still caught.

    Confirmed directly for triangular (2,2): the raw wrapped node pairs
    include both (0, 2) and (2, 0) as separate ordered instances of the
    same undirected collision -- a naive ``set[tuple[int, int]]`` without
    canonicalizing to (min, max) would miss this entirely. Both the real
    graph builder (via ``nx.Graph.has_edge``, inherently undirected) and
    the standalone matrix-validity simulation (via ``_canonical_edge``)
    must still detect it.
    """
    tiling = registry.get("triangular")
    with pytest.raises(ValueError, match="alias|multiplicity"):
        tiling.graph((2, 2))
    assert not _matrix_supercell_is_valid(
        tiling, np.array([[2, 0], [0, 2]], dtype=np.intp)
    )
