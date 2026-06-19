"""Snub square tiling — vertex configuration 3².4.3.4.

Each vertex borders two pairs of triangles and one square, in alternation.
Primitive cell: 4 vertices.  Coordination number: 5.  Non-bipartite.
Lattice: square (a = b, γ = 90°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register

# For the exact 2D snub square tiling with regular polygons,
# the lattice parameter is precisely derived as a = bond * √(2 + √3).
_S = np.sqrt(3.0)
_SCALE = np.sqrt(2.0 + _S)


@register("3.3.4.3.4", "3².4.3.4", "snub_square")
class SnubSquareTiling(Tiling):
    """Snub square tiling — 3².4.3.4."""

    @property
    def vertex_config(self) -> str:
        return "3².4.3.4"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        a = self.bond_length * _SCALE
        # Square Bravais lattice.
        return np.array(
            [
                [a, 0.0],
                [0.0, a],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # 4 vertices per primitive cell.
        # The square is rotated by exactly 15° relative to the lattice.
        # (Standard square at 45° + 15° rotation = 60° starting angle).
        # The circumradius of a square with side bond_length is bond_length / √2
        r = self.bond_length / np.sqrt(2.0)
        angles = np.radians(np.array([60.0, 150.0, 240.0, 330.0]))

        cart = np.array(
            [[r * np.cos(a), r * np.sin(a)] for a in angles],
            dtype=np.float64,
        )

        # Convert Cartesian coordinates to Fractional coordinates
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

        # Dynamically loop over all 4 vertices to map the coordination of 5
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
        return False
