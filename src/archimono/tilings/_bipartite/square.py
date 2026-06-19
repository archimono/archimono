"""Square tiling — vertex configuration 4⁴.

Each vertex borders four squares.
Primitive cell: 2 vertices.  Coordination number: 4.  Bipartite.
Lattice: square (a = b, γ = 90°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register


@register("4^4", "4⁴", "square")
class SquareTiling(Tiling):
    """Square tiling — 4⁴."""

    @property
    def vertex_config(self) -> str:
        return "4⁴"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # To fit 2 atoms (bipartite checkerboard) while maintaining exact bond lengths,
        # the square unit cell spans the diagonal of the individual squares.
        # Therefore, a = bond * √2
        a = self.bond_length * np.sqrt(2.0)
        return np.array(
            [
                [a, 0.0],
                [0.0, a],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # Bipartite checkerboard: Sublattice A at origin, Sublattice B at center
        return np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)

    @property
    def edges(self) -> list[tuple[int, int, int, int]]:
        # Vertex 0 connects exactly to 4 Vertex 1s across the lattice.
        # Total edges = 4, satisfying (2 vertices * degree 4) / 2 = 4 unique edges.
        return [
            (0, 1, 0, 0),    # Top-Right
            (0, 1, -1, 0),   # Top-Left
            (0, 1, 0, -1),   # Bottom-Right
            (0, 1, -1, -1),  # Bottom-Left
        ]

    @property
    def is_bipartite(self) -> bool:
        return True
