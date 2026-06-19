"""Orderly generation of symmetry-inequivalent binary colorings.

Enumerates exactly one canonical (lexicographically smallest) representative
per symmetry orbit of balanced binary assignments on a periodic lattice.
The algorithm builds assignments site-by-site and prunes branches whose
partial prefix is already lexicographically larger than some group-image
prefix, guaranteeing that only canonical representatives survive to the
leaves of the search tree.

The early-termination bound comes from Burnside's lemma: the number of
orbits is computed in ``O(|G| · n²)`` via the cycle-length knapsack DP in
:class:`~archimono.assignment.enumeration.burnside.BurnsideCounter`, so the search
stops as soon as that many representatives have been collected.

References:
    Read, R. C. (1978).
        Every one a winner; or, how to avoid isomorphism search when
        cataloguing combinatorial configurations.
        *Annals of Discrete Mathematics*, 2, 107–120.
        https://doi.org/10.1016/S0167-5060(08)70325-X

    McKay, B. D. (1998).
        Isomorph-free exhaustive generation.
        *Journal of Algorithms*, 26(2), 306–324.
        https://doi.org/10.1006/jagm.1997.0898
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ase import Atoms

from archimono.assignment.enumeration._helpers import (
    _get_symmetry_permutations,
    _tiling_to_atoms,
    assignment_to_atoms,
)
from archimono.assignment.enumeration.burnside import BurnsideCounter
from archimono.assignment.enumeration.canonical import filter_active, prepare_active

if TYPE_CHECKING:
    from archimono.tilings.base import SupercellLike, Tiling


class _EarlyTermination(Exception):
    """Raised internally when all orbits have been found."""


class OrderlyEnumerator:
    """Enumerate symmetry-inequivalent binary assignments via orderly generation.

    Uses backtracking with partial canonical pruning to generate exactly
    one lex-minimum representative per symmetry orbit.  Does not depend on
    icet; uses spglib (via
    :func:`~archimono.assignment.enumeration._helpers._get_symmetry_permutations`)
    for the symmetry group and :meth:`BurnsideCounter.count` for the
    early-termination bound.

    The algorithm assigns sites left-to-right (0, 1, …, n−1), trying
    value 0 before 1 at each position.  At each step
    :func:`~archimono.assignment.enumeration.canonical.filter_active` prunes the
    active set of group elements and detects non-canonical prefixes.

    References:
        Read (1978); McKay (1998): see module docstring.

    Example::

        >>> from archimono.tilings import registry
        >>> from archimono.assignment import OrderlyEnumerator
        >>>
        >>> tiling = registry.get("kagome")
        >>> enumerator = OrderlyEnumerator()
        >>> configs = enumerator.enumerate(tiling, n_b=6, supercell=(2, 2))
        >>> len(configs)
        30
    """

    @staticmethod
    def generate_canonical(
        n: int,
        n_b: int,
        group: list[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """Generate all canonical (lex-minimum) orbit representatives.

        Core orderly generation algorithm, independent of any tiling or
        lattice.  Operates on abstract site indices and a permutation
        group.

        Args:
            n: Total number of sites.
            n_b: Number of label-1 atoms (= species[1]).  Must
                satisfy ``0 <= n_b <= n``.
            group: Symmetry group as a list of permutation tuples of
                length *n*.  Must contain the identity.

        Returns:
            One canonical assignment per orbit, in lexicographic
            order.  Each tuple has length *n* with exactly *n_b*
            ones.

        Raises:
            ValueError: If *n_b* is out of range or *group* is
                invalid (see :meth:`BurnsideCounter.count`).
        """
        if not (0 <= n_b <= n):
            raise ValueError(f"n_b={n_b} out of valid range [0, {n}].")

        expected = BurnsideCounter.count(n, n_b, group)

        initial_active = prepare_active(group)
        results: list[tuple[int, ...]] = []
        assignment: list[int] = []

        def _backtrack(
            pos: int,
            ones_left: int,
            zeros_left: int,
            active: list[tuple[int, ...]],
        ) -> None:
            if pos == n:
                if ones_left == 0:
                    results.append(tuple(assignment))
                    if len(results) == expected:
                        raise _EarlyTermination
                return

            for value in (0, 1):
                if value == 0 and zeros_left <= 0:
                    continue
                if value == 1 and ones_left <= 0:
                    continue

                new_active, is_canonical = filter_active(
                    active, assignment, pos, value,
                )
                if not is_canonical:
                    continue

                assignment.append(value)
                _backtrack(
                    pos + 1,
                    ones_left - value,
                    zeros_left - (1 - value),
                    new_active,
                )
                assignment.pop()

        try:
            _backtrack(0, n_b, n - n_b, initial_active)
        except _EarlyTermination:
            pass

        return results

    def enumerate(
        self,
        tiling: Tiling,
        n_b: int,
        supercell: SupercellLike = (1, 1),
        min_cut: int = 0,
        *,
        override: bool = False,
    ) -> list[tuple[int, ...]]:
        """Enumerate all symmetry-inequivalent balanced binary assignments.

        Args:
            tiling: The Archimedean tiling to enumerate on.
            n_b: Number of label-1 atoms (= species[1]).  Must satisfy
                ``1 <= n_b < n``.
            supercell: ``(na, nb)`` tuple or 2×2 integer matrix supercell
                (matching :meth:`Tiling.graph`).
            min_cut: Include only configurations with cut value
                >= *min_cut*.
            override: If ``True``, skip the combinatorial-explosion
                safety check.

        Returns:
            One canonical assignment tuple per symmetry-inequivalent
            orbit.  Each tuple is a length-*n* binary sequence
            (0 = ``species[0]``, 1 = ``species[1]``).  Atom ordering matches
            the node ordering of
            :meth:`~archimono.tilings.base.Tiling.graph`.

        Raises:
            ValueError: If *supercell* produces a degenerate graph
                (self-loop or PBC parity violation), if *n_b* is
                out of range, or if the raw enumeration space
                C(n, n_b) exceeds 10 million without
                ``override=True``.
        """
        graph = tiling.graph(supercell)
        atoms = _tiling_to_atoms(tiling, supercell)
        n = len(atoms)

        if not (1 <= n_b < n):
            raise ValueError(
                f"n_b={n_b} out of valid range [1, {n - 1}] for n={n}."
            )

        _MAX_RAW = 10_000_000
        n_choose_nb = math.comb(n, n_b)
        if not override and n_choose_nb > _MAX_RAW:
            raise ValueError(
                f"Raw enumeration space C({n}, {n_b}) = {n_choose_nb:,} "
                f"exceeds the {_MAX_RAW:,}-structure safety limit.  This "
                f"will likely exhaust memory or run indefinitely.  Pass "
                f"override=True only if you have confirmed the enumeration "
                f"is tractable."
            )

        group = _get_symmetry_permutations(atoms)
        assignments = self.generate_canonical(n, n_b, group)

        if min_cut > 0:
            adj: list[list[int]] = [[] for _ in range(n)]
            for u, v in graph.edges():
                adj[u].append(v)
                adj[v].append(u)

            def cut(a: tuple[int, ...]) -> int:
                return sum(
                    1
                    for u in range(n)
                    for v in adj[u]
                    if v > u and a[u] != a[v]
                )

            assignments = [a for a in assignments if cut(a) >= min_cut]

        return assignments

    def get_symmetry_group(
        self,
        tiling: Tiling,
        supercell: SupercellLike = (1, 1),
    ) -> list[tuple[int, ...]]:
        """Return the crystallographic symmetry group as site permutations.

        Args:
            tiling: Target tiling.
            supercell: ``(na, nb)`` supercell dimensions.

        Returns:
            Unique permutation tuples representing the space-group
            symmetry.
        """
        atoms = _tiling_to_atoms(tiling, supercell)
        return _get_symmetry_permutations(atoms)

    @staticmethod
    def to_atoms(
        tiling: Tiling,
        supercell: SupercellLike,
        assignment: tuple[int, ...],
        species: tuple[str, str] = ("B", "N"),
    ) -> Atoms:
        """Convert a binary assignment to an ASE Atoms structure.

        Args:
            tiling: The Archimedean tiling the assignment was
                generated for.
            supercell: ``(na, nb)`` tuple or 2×2 integer matrix supercell
                used during enumeration.
            assignment: A binary tuple of length *n*
                (0 = ``species[0]``, 1 = ``species[1]``).
            species: Pair of ASE-valid element symbols.

        Returns:
            An ASE Atoms object with the species assigned.

        Raises:
            ValueError: If ``len(assignment)`` does not match the supercell
                atom count.
        """
        return assignment_to_atoms(tiling, supercell, assignment, species)
