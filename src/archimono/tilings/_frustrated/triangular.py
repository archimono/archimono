"""Triangular tiling — vertex configuration 3⁶.

Each vertex borders six triangles.
Primitive cell: 1 vertex.  Coordination number: 6.  Non-bipartite.
Lattice: hexagonal (a = b, γ = 60°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register


@register("3^6", "3⁶", "triangular")
class TriangularTiling(Tiling):
    """Triangular tiling — 3⁶."""

    @property
    def vertex_config(self) -> str:
        return "3⁶"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # Standard hexagonal lattice
        a = self.bond_length
        return np.array(
            [
                [a, 0.0],
                [a / 2.0, a * np.sqrt(3) / 2.0],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # 1 vertex per primitive cell at the origin
        return np.array([[0.0, 0.0]], dtype=np.float64)

    @property
    def edges(self) -> list[tuple[int, int, int, int]]:
        # 3 unique undirected edges per primitive cell to achieve degree 6.
        # The other 3 neighbors are reached when adjacent cells cast their
        # edges backward.
        return [
            (0, 0, 1, 0),   # Right
            (0, 0, 0, 1),   # Top-Right
            (0, 0, 1, -1),  # Bottom-Right
        ]

    @property
    def is_bipartite(self) -> bool:
        return False
