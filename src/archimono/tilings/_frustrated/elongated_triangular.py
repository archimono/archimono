"""Elongated triangular tiling — vertex configuration 3³.4².

Each vertex borders three triangles and two squares in alternation.
Primitive cell: 4 vertices.  Coordination number: 5.  Non-bipartite.
Lattice: rectangular (a ≠ b, γ = 90°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register

_S = np.sqrt(3)


@register("3.3.3.4.4", "3³.4²", "elongated_triangular")
class ElongatedTriangularTiling(Tiling):
    """Elongated triangular tiling — 3³.4².

    Note:
        PBC self-loop constraint — the supercell must have **na ≥ 2**.  The
        primitive edge list includes edges of the form ``(i, i, -1, 0)``
        (same vertex type across the a-direction boundary).  With na=1, PBC
        wraps ``(ia−1) % 1 = ia``, collapsing these into self-loops.
        ``Tiling.graph()`` raises ``ValueError`` for na=1.  See
        ``docs/reference/tilings.md``, §PBC.
    """

    @property
    def vertex_config(self) -> str:
        return "3³.4²"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # A true rectangular cell spans two rows of squares and two rows of triangles
        # to account for the alternating 0.5 x-shift.
        # b = 2 * (square_height) + 2 * (triangle_height)
        # b = 2 * bond + 2 * (bond * √3 / 2) = bond * (2 + √3)
        a = self.bond_length
        b = self.bond_length * (2.0 + _S)
        return np.array(
            [
                [a, 0.0],
                [0.0, b],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # 4 vertices per rectangular cell.
        # We calculate their fractional y-positions based on the total height factor D.
        D = 2.0 + _S
        y0 = 0.0                        # Base of the 1st square
        y1 = 1.0 / D                    # Top of the 1st square
        y2 = 0.5                        # Base of the 2nd square (shifted by 0.5 in x)
        y3 = (2.0 + _S / 2.0) / D       # Top of the 2nd square

        return np.array(
            [
                [0.0, y0],
                [0.0, y1],
                [0.5, y2],
                [0.5, y3],
            ],
            dtype=np.float64,
        )

    @property
    def edges(self) -> list[tuple[int, int, int, int]]:
        # 10 unique undirected edges per primitive cell → degree 5 at each vertex.
        return [
            # Square edges (horizontal, same-vertex across a-boundary)
            (0, 0, 1, 0),
            (1, 1, 1, 0),
            (2, 2, 1, 0),
            (3, 3, 1, 0),
            # Square edges (vertical, intra-cell)
            (0, 1, 0, 0),
            (2, 3, 0, 0),
            # Triangle edges (diagonal, connecting square rows)
            (1, 2, 0, 0),
            (1, 2, -1, 0),
            (0, 3, 0, -1),
            (0, 3, -1, -1),
        ]

    @property
    def is_bipartite(self) -> bool:
        return False
