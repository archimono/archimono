"""Hexagonal tiling — vertex configuration 6³.

Each vertex borders three hexagons.
Primitive cell: 2 vertices.  Coordination number: 3.  Bipartite.
Lattice: hexagonal (a = b, γ = 120°).
Bond length is parameterised via ``Tiling.bond_length`` (default 1.0).
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling
from archimono.tilings.registry import register


@register("6^3", "6³", "hexagonal")
class HexagonalTiling(Tiling):
    """Hexagonal tiling — 6³."""

    @property
    def vertex_config(self) -> str:
        return "6³"

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        # a1 = a(1, 0),  a2 = a(-1/2, √3/2)  with a = bond * √3
        # This forms the standard 120° lattice
        a = self.bond_length * np.sqrt(3.0)
        return np.array(
            [
                [a, 0.0],
                [-a / 2.0, a * np.sqrt(3.0) / 2.0],
            ],
            dtype=np.float64,
        )

    @property
    def vertices(self) -> NDArray[np.float64]:
        # Two atoms (A and B sublattice) in fractional coordinates.
        # For a 120° lattice, the correct placement is (1/3, 2/3) and (2/3, 1/3)
        return np.array(
            [[1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0]],
            dtype=np.float64
        )

    @functools.cached_property
    def edges(self) -> list[tuple[int, int, int, int]]:
        frac = self.vertices
        L = self.lattice_vectors
        offsets = [(da, db) for da in (-1, 0, 1) for db in (-1, 0, 1)]

        edges: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        tol = 0.15 * self.bond_length

        # Dynamically discover all 3 connections per vertex to satisfy
        # the honeycomb structure.
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
