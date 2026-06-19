"""Truncated trihexagonal tiling — vertex configuration 4.6.12.

Each vertex borders one square, one hexagon, and one 12-gon.
Primitive cell: 12 vertices.  Coordination number: 3.  Bipartite.
Lattice: hexagonal (a = b, γ = 120°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register

_S = np.sqrt(3.0)


@register("4.6.12", "truncated_trihexagonal")
class TruncatedTrihexagonalTiling(Tiling):
    """Truncated trihexagonal tiling — 4.6.12."""

    @property
    def vertex_config(self) -> str:
        return "4.6.12"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # The true primitive cell spans the distance between the centers
        # of adjacent 12-gons.
        # This distance exactly equals bond * (3 + √3).
        a = self.bond_length * (3.0 + _S)
        return np.array(
            [
                [a, 0.0],
                [-a / 2.0, a * _S / 2.0],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # 12 vertices per primitive cell.
        # By centering a single 12-gon at the origin, we capture exactly all
        # 12 unique vertices for the primitive cell without any manual piecing
        # of hexagons/squares!
        angles = np.radians(np.arange(12) * 30.0 + 15.0)

        # Exact circumradius of a 12-gon with side length bond_length
        R = self.bond_length * np.sqrt(2.0 + _S)

        cart = np.array(
            [[R * np.cos(a), R * np.sin(a)] for a in angles],
            dtype=np.float64,
        )

        # Convert Cartesian to Fractional coordinates
        L = self.lattice_vectors
        frac = np.linalg.solve(L.T, cart.T).T.astype(np.float64)
        return frac % 1.0

    @property
    def edges(self) -> list[tuple[int, int, int, int]]:
        frac = self.vertices
        L = self.lattice_vectors
        offsets = [(da, db) for da in (-1, 0, 1) for db in (-1, 0, 1)]

        edges: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        tol = 0.15 * self.bond_length

        # Dynamically discover all 18 undirected edges per primitive cell
        # to perfectly achieve the coordination number of 3 for all 12 vertices.
        for i in range(len(frac)):
            for j in range(len(frac)):
                for da, db in offsets:
                    if i == j and da == 0 and db == 0:
                        continue
                    shifted = frac[j] + np.array([da, db])
                    dist = float(np.linalg.norm((shifted - frac[i]) @ L))
                    if abs(dist - self.bond_length) < tol:
                        key = (i, j, da, db)
                        rev = (j, i, -da, -db)
                        if key not in seen and rev not in seen:
                            edges.append(key)
                            seen.add(key)
        return edges

    @property
    def is_bipartite(self) -> bool:
        return True
