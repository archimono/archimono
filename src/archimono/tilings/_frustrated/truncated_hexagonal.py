"""Truncated hexagonal tiling — vertex configuration 3.12².

Each vertex borders one triangle and two 12-gons.
Primitive cell: 6 vertices.  Coordination number: 3.  Non-bipartite.
Lattice: hexagonal (a = b, γ = 120°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register

_S = np.sqrt(3.0)


@register("3.12.12", "3.12²", "truncated_hexagonal")
class TruncatedHexagonalTiling(Tiling):
    """Truncated hexagonal tiling — 3.12²."""

    @property
    def vertex_config(self) -> str:
        return "3.12²"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # Distance between centers of adjacent 12-gons
        a = self.bond_length * (2.0 + _S)
        return np.array(
            [
                [a, 0.0],
                [-a / 2.0, a * _S / 2.0],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # 6 vertices per primitive cell.
        # We place a 12-gon at the origin. Because each vertex is shared by two 12-gons
        # across the unit cells, its 12 vertices resolve to exactly 6 unique
        # fractional points.  We use 6 distinct consecutive angles of the
        # 12-gon to form our asymmetric unit:
        angles = np.radians([-15.0, 15.0, 45.0, 75.0, 105.0, 135.0])

        # Circumradius of a 12-gon with edge length bond_length
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

        # Dynamically discover all 9 undirected edges per primitive cell
        # to perfectly achieve the coordination number of 3 for all 6 vertices.
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
