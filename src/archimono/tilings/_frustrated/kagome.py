"""Kagome (trihexagonal) tiling — vertex configuration 3.6.3.6.

Each vertex borders two triangles and two hexagons.
Primitive cell: 3 vertices.  Coordination number: 4.  Non-bipartite.
Lattice: hexagonal (a = b, γ = 60°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register


@register("3.6.3.6", "kagome")
class KagomeTiling(Tiling):
    """Kagome (trihexagonal) tiling — 3.6.3.6."""

    @property
    def vertex_config(self) -> str:
        return "3.6.3.6"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # Kagome has the same hexagonal lattice as honeycomb but a = 2*bond.
        a = 2.0 * self.bond_length
        return np.array(
            [
                [a, 0.0],
                [a / 2.0, a * np.sqrt(3) / 2.0],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # Three vertices at edge midpoints of the hexagonal cell:
        #   v0 = (1/2, 0),  v1 = (0, 1/2),  v2 = (1/2, 1/2)
        return np.array(
            [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
            dtype=np.float64,
        )

    @property
    def edges(self) -> list[tuple[int, int, int, int]]:
        # Each vertex has degree 4.
        # Triangle edges (within cell):
        return [
            (0, 1, 0, 0),
            (0, 2, 0, 0),
            (1, 2, 0, 0),

            # Cross-cell edges completing the coordination:
            # Corrected: v0 to v1 in cell (1, -1) maintains exact _BOND distance
            (0, 1, 1, -1),
            (0, 2, 0, -1),
            (1, 2, -1, 0),
        ]

    @property
    def is_bipartite(self) -> bool:
        return False
