"""Rhombitrihexagonal tiling — vertex configuration 3.4.6.4.

Each vertex borders one triangle, two squares, and one hexagon.
Primitive cell: 6 vertices.  Coordination number: 4.  Non-bipartite.
Lattice: hexagonal (a = b, γ = 120°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register

_S = np.sqrt(3)


@register("3.4.6.4", "rhombitrihexagonal")
class RhombitrihexagonalTiling(Tiling):
    """Rhombitrihexagonal tiling — 3.4.6.4."""

    @property
    def vertex_config(self) -> str:
        return "3.4.6.4"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # Hexagonal lattice: distance between hexagon centers across a square
        # a = 2*(apothem of hexagon) + (side of square) = bond*√3 + bond
        a = self.bond_length * (1.0 + _S)
        return np.array(
            [
                [a, 0.0],
                [-a / 2.0, a * _S / 2.0],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # 6 vertices per primitive cell forming a regular hexagon.
        # The radius of a regular hexagon is exactly its side length (bond_length).
        # We rotate by 30° so the flat edges face the lattice vectors,
        # leaving exact room for the connecting squares.
        angles_hex = np.radians(np.arange(6) * 60.0 + 30.0)

        cart = np.array(
            [
                [self.bond_length * np.cos(a), self.bond_length * np.sin(a)]
                for a in angles_hex
            ],
            dtype=np.float64,
        )
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

        # Dynamically discover all edges (hexagon ring + squares + triangles)
        # This guarantees the coordination number of 4 is met without
        # hardcoding boundaries.
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
