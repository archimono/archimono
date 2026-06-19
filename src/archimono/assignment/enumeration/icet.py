"""Symmetry-inequivalent structure enumeration using icet.

``IcetEnumerator`` wraps icet's ``enumerate_structures`` function to produce
all symmetry-inequivalent binary-species assignments for a given tiling
supercell.  The icet library uses spglib internally for crystallographic
symmetry reduction, and its derivative-structure enumeration algorithm
(Hart & Forcade, 2008) guarantees completeness.

Note:
    ``enumerate_structures`` is called with the **supercell** (not the
    primitive cell) as the reference structure and ``sizes=[1]``.  This
    restricts enumeration to the specific supercell geometry requested,
    rather than enumerating all supercell shapes of the same atom count.
    Passing the primitive cell with ``sizes=[na*nb]`` would include all
    other supercell shapes of the same volume, producing more structures
    than intended.

References:
    icet structure enumeration:
        Ångqvist, M., Muñoz, W. A., Rahm, J. M., Fransson, E.,
        Durniak, C., Rozyczko, P., … Erhart, P. (2019).
        ICET — A Python library for constructing and sampling alloy
        cluster expansions.
        *Advanced Theory and Simulations*, 2(7), 1900015.
        https://doi.org/10.1002/adts.201900015

    Hart–Forcade derivative-structure enumeration algorithm:
        Hart, G. L. W. & Forcade, R. W. (2008).
        Algorithm for generating derivative structures.
        *Physical Review B*, 77(22), 224115.
        https://doi.org/10.1103/PhysRevB.77.224115

    spglib (used internally by icet):
        Togo, A., Shinohara, K. & Tanaka, I. (2018).
        Spglib: a software library for crystal symmetry search.
        arXiv:1808.01590.
        https://arxiv.org/abs/1808.01590
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

if TYPE_CHECKING:
    from archimono.tilings.base import SupercellLike, Tiling


class IcetEnumerator:
    """Enumerate symmetry-inequivalent balanced binary assignments using icet.

    Wraps ``icet.tools.enumerate_structures`` with the tiling supercell as the
    reference structure (``sizes=[1]``).  This restricts enumeration to the
    specific supercell geometry and avoids generating structures from other
    supercell shapes of the same volume.

    icet uses the Hart–Forcade derivative-structure enumeration algorithm
    internally, which guarantees completeness: every symmetry-inequivalent
    structure is generated exactly once.

    Example::

        >>> from archimono.tilings import registry
        >>> from archimono.assignment import BruteforceSolver, IcetEnumerator
        >>>
        >>> tiling = registry.get("kagome")
        >>> graph   = tiling.graph(supercell=(2, 2))
        >>> result  = BruteforceSolver().solve(graph, ["A", "B"], n_b=6)
        >>> print(result.cut_value)         # 16
        >>>
        >>> enumerator = IcetEnumerator()
        >>> configs    = enumerator.enumerate(tiling, n_b=6, supercell=(2, 2))
        >>> print(len(configs))             # 30
        >>>
        >>> configs_near_opt = enumerator.enumerate(
        ...     tiling, n_b=6, supercell=(2, 2),
        ...     min_cut=result.cut_value - 2,
        ... )
        >>> print(len(configs_near_opt))    # 15

    References:
        Ångqvist et al. (2019); Hart & Forcade (2008): see module
        docstring.
    """

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
                (matching :meth:`Tiling.graph`).  The supercell atoms are
                used directly as the reference structure for icet.
            min_cut: Include only configurations with cut value
                >= *min_cut*.  Set to
                ``solver_result.cut_value - tolerance`` to restrict
                to structures within *tolerance* bonds of the
                optimum.
            override: If ``True``, skip the combinatorial-explosion
                safety check.  Use only when you have confirmed the
                enumeration is tractable.

        Returns:
            One assignment tuple per symmetry-inequivalent orbit.
            Each tuple is a length-*n* binary sequence
            (0 = ``species[0]``, 1 = ``species[1]``).  Atom ordering matches
            the node ordering of
            :meth:`~archimono.tilings.base.Tiling.graph`.

        Raises:
            ValueError: If *supercell* produces a degenerate graph
                (self-loop or PBC parity violation), if *n_b* is
                out of range, or if the raw enumeration space
                C(n, n_b) exceeds 10 million without
                ``override=True``.
            RuntimeError: If icet's ``enumerate_structures`` returns
                no structures (indicates a geometry or concentration
                issue).
            ImportError: If the optional ``icet`` dependency is not
                installed.  The icet-free orderly backend
                (``enumeration.get("orderly")``) has no such
                requirement.
        """
        # Imported lazily so the package (and the default orderly backend)
        # stays importable without the optional ``icet`` dependency.
        from icet.tools import enumerate_structures  # type: ignore[import-untyped]

        graph = tiling.graph(supercell)
        atoms = _tiling_to_atoms(tiling, supercell)
        n = len(atoms)

        if not (1 <= n_b < n):
            raise ValueError(f"n_b={n_b} out of valid range [1, {n - 1}] for n={n}.")

        _MAX_RAW = 10_000_000
        n_choose_nb = math.comb(n, n_b)
        if not override and n_choose_nb > _MAX_RAW:
            raise ValueError(
                f"Raw enumeration space C({n}, {n_b}) = {n_choose_nb:,} exceeds "
                f"the {_MAX_RAW:,}-structure safety limit.  This will likely "
                f"exhaust memory or run indefinitely.  Pass override=True only "
                f"if you have confirmed the enumeration is tractable."
            )

        conc = n_b / n

        # icet requires ASE-valid element symbols to build reference structures,
        # but the integer-label output is invariant to that choice: any two
        # distinct valid symbols produce identical orbits.  A fixed binary pair
        # therefore keeps the public signature backend-agnostic (matching
        # OrderlyEnumerator) without affecting results.
        species = ("B", "N")

        raw_structs = list(
            enumerate_structures(
                structure=atoms,
                sizes=[1],
                chemical_symbols=[list(species)] * n,
                concentration_restrictions={species[1]: (conc, conc)},
            )
        )

        if not raw_structs:
            raise RuntimeError(
                "icet returned no structures.  Check that the concentration "
                f"n_b/n = {n_b}/{n} = {conc:.4f} is achievable for this supercell."
            )

        assignments: list[tuple[int, ...]] = [
            tuple(1 if s == species[1] else 0 for s in struct.get_chemical_symbols())
            for struct in raw_structs
        ]

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

    @staticmethod
    def to_atoms(
        tiling: Tiling,
        supercell: SupercellLike,
        assignment: tuple[int, ...],
        species: tuple[str, str] = ("B", "N"),
    ) -> Atoms:
        """Convert a binary assignment to an ASE Atoms structure.

        Args:
            tiling: The Archimedean tiling the assignment was generated for.
            supercell: ``(na, nb)`` tuple or 2×2 integer matrix supercell used
                during enumeration.
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

    def get_symmetry_group(
        self,
        tiling: Tiling,
        supercell: SupercellLike = (1, 1),
    ) -> list[tuple[int, ...]]:
        """Return the crystallographic symmetry group as site permutations.

        Uses spglib directly (the same backend as icet) to extract site
        permutations for the tiling supercell.  Useful for computing
        Burnside counts independently of the icet enumeration.

        Args:
            tiling: Target tiling.
            supercell: ``(na, nb)`` supercell dimensions.

        Returns:
            Unique permutation tuples representing the space-group
            symmetry.
        """
        atoms = _tiling_to_atoms(tiling, supercell)
        return _get_symmetry_permutations(atoms)
