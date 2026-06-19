"""Truncated square tiling — vertex configuration 4.8².

Each vertex borders one square and two octagons.
Primitive cell: 4 vertices.  Coordination number: 3.  Bipartite.
Lattice: square (a = b, γ = 90°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register


@register("4.8.8", "4.8²", "truncated_square")
class TruncatedSquareTiling(Tiling):
    """Truncated square tiling — 4.8².

    Note:
        PBC parity constraint — both supercell dimensions (na, nb) must be
        **even**.  Odd dimensions fold a cross-cell edge connecting two
        same-sublattice vertices into the same cell, producing an A–A bond that
        destroys bipartiteness.  ``Tiling.graph()`` raises ``ValueError`` for
        odd-dimension supercells.  See ``docs/reference/tilings.md``, §PBC.
    """

    @property
    def vertex_config(self) -> str:
        return "4.8²"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # Lattice parameter: a = bond * (1 + √2)
        a = self.bond_length * (1.0 + np.sqrt(2.0))
        return np.array(
            [
                [a, 0.0],
                [0.0, a],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # Four vertices per primitive cell.
        # We place the square perfectly in the center of the unit cell (0.5, 0.5).
        # It is naturally rotated by 45 degrees to bridge the octagons
        # located at the cell corners.
        a = self.bond_length * (1.0 + np.sqrt(2.0))

        # Calculate the fractional distance from the center to the square's corners
        d = self.bond_length / (2.0 * a)

        return np.array(
            [
                [d, 0.5],           # Left vertex of the square
                [0.5, d],           # Bottom vertex of the square
                [1.0 - d, 0.5],     # Right vertex of the square
                [0.5, 1.0 - d],     # Top vertex of the square
            ],
            dtype=np.float64,
        )

    @functools.cached_property
    def edges(self) -> list[tuple[int, int, int, int]]:
        frac = self.vertices
        L = self.lattice_vectors
        offsets = [(da, db) for da in (-1, 0, 1) for db in (-1, 0, 1)]

        edges: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        tol = 0.15 * self.bond_length

        # Dynamically discover all 6 undirected edges per primitive cell
        # to perfectly achieve the coordination number of 3 for all 4 vertices.
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
