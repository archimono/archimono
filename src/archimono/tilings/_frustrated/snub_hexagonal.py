"""Snub hexagonal tiling — vertex configuration 3⁴.6.

Each vertex borders four triangles and one hexagon.
Primitive cell: 6 vertices.  Coordination number: 5.  Non-bipartite.
Lattice: hexagonal (a = b, γ = 60°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register

_S = np.sqrt(3)


@register("3.3.3.3.6", "3⁴.6", "snub_hexagonal")
class SnubHexagonalTiling(Tiling):
    """Snub hexagonal tiling — 3⁴.6."""

    @property
    def vertex_config(self) -> str:
        return "3⁴.6"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        a = self.bond_length * np.sqrt(7.0)
        return np.array(
            [
                [a, 0.0],
                [a / 2.0, a * _S / 2.0],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # Rotation angle θ such that a hexagon vertex at (cos θ, sin θ)
        # is at bond distance from the nearest vertex in the (1, 0) neighbouring
        # cell (lattice parameter a = bond × √7).
        # Constraint: |[√7, 0] − 2(cos θ, sin θ)|² = 1
        # Expanding: 7 − 4√7 cos θ + 4 = 1  →  cos θ = 5/(2√7)
        # Therefore θ = arctan(√3 / 5).
        theta = np.arctan(_S / 5.0)

        # Calculate angles for the 6 vertices of the hexagon
        angles_hex = np.radians(np.arange(6) * 60.0) + theta

        # Generate Cartesian coordinates (centered at origin)
        cart = np.array(
            [
                [self.bond_length * np.cos(a), self.bond_length * np.sin(a)]
                for a in angles_hex
            ],
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

        # Dynamically loop over all 6 vertices to find the coordination of 5
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
