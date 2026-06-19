"""Shared tiling/assignment-to-atoms and symmetry-extraction utilities.

Used by both :class:`~archimono.assignment.enumeration.icet.IcetEnumerator`
and :class:`~archimono.assignment.enumeration.orderly.OrderlyEnumerator`, and
by :mod:`archimono.export` for converting enumerated assignments to structures.

References:
    Togo, A., Shinohara, K. & Tanaka, I. (2018).
        Spglib: a software library for crystal symmetry search.
        arXiv:1808.01590.
        https://arxiv.org/abs/1808.01590
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import spglib
from ase import Atoms

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from archimono.tilings.base import SupercellLike, Tiling


def _supercell_lattice_vectors(
    supercell: SupercellLike,
    a1: NDArray[np.float64],
    a2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the two Cartesian supercell lattice vectors ``(A1, A2)``.

    For a diagonal ``(na, nb)`` tuple the supercell vectors are ``na*a1`` and
    ``nb*a2``.  For a 2×2 integer matrix ``S`` they follow the same convention
    as :meth:`~archimono.tilings.base.Tiling.graph`: ``A_i = sum_j S_ji a_j``,
    i.e. column ``i`` of ``S`` gives the coefficients of ``A_i``.

    Args:
        supercell: Either an ``(na, nb)`` tuple or a 2×2 integer matrix.
        a1: First primitive lattice vector (2D).
        a2: Second primitive lattice vector (2D).

    Returns:
        The pair of 2D Cartesian supercell lattice vectors.
    """
    arr = np.asarray(supercell)
    if arr.ndim == 2:
        s = arr.astype(np.float64)
        return s[0, 0] * a1 + s[1, 0] * a2, s[0, 1] * a1 + s[1, 1] * a2
    na, nb = int(arr[0]), int(arr[1])
    return na * a1, nb * a2


def _tiling_to_atoms(
    tiling: Tiling,
    supercell: SupercellLike,
    vacuum: float = 20.0,
) -> Atoms:
    """Convert a :class:`~archimono.tilings.base.Tiling` to an ASE Atoms object.

    Atom ``i`` corresponds exactly to node ``i`` of
    :meth:`~archimono.tilings.base.Tiling.graph` — positions are read straight
    from the graph's ``pos`` attribute so the atom order matches the graph node
    order for both diagonal ``(na, nb)`` tuple supercells and 2×2 integer matrix
    (sheared) supercells.  This ordering is preserved by icet's
    ``enumerate_structures`` when the supercell atoms are passed as the
    reference structure with ``sizes=[1]``.

    All atoms are assigned hydrogen (``'H'``) as a placeholder species.
    PBC is ``[True, True, False]``; the slab is centred at ``z = vacuum / 2``.

    Args:
        tiling: The tiling to convert.
        supercell: Either an ``(na, nb)`` tuple or a 2×2 integer matrix
            (sheared) supercell, matching :meth:`Tiling.graph`.
        vacuum: Out-of-plane cell height in Å.

    Returns:
        An ASE :class:`~ase.Atoms` object with one atom per graph node.
    """
    a1_2d = np.asarray(tiling.lattice_vectors[0], dtype=np.float64)
    a2_2d = np.asarray(tiling.lattice_vectors[1], dtype=np.float64)

    A1, A2 = _supercell_lattice_vectors(supercell, a1_2d, a2_2d)
    cell = np.array(
        [
            [A1[0], A1[1], 0.0],
            [A2[0], A2[1], 0.0],
            [0.0, 0.0, vacuum],
        ]
    )

    # Read positions from the graph so atom i == graph node i for every
    # supercell shape (the assignment tuple is indexed by graph node id).
    graph = tiling.graph(supercell)
    n = graph.number_of_nodes()
    positions = [
        [
            float(graph.nodes[i]["pos"][0]),
            float(graph.nodes[i]["pos"][1]),
            vacuum / 2.0,
        ]
        for i in range(n)
    ]

    return Atoms(
        symbols=["H"] * n,
        positions=positions,
        cell=cell,
        pbc=[True, True, False],
    )


def assignment_to_atoms(
    tiling: Tiling,
    supercell: SupercellLike,
    assignment: tuple[int, ...],
    species: tuple[str, str] = ("B", "N"),
) -> Atoms:
    """Convert a binary assignment tuple to an ASE Atoms structure.

    Builds the full supercell geometry for *tiling* and *supercell* (via
    :func:`_tiling_to_atoms`), then assigns *species* according to the binary
    *assignment*.  This is the single canonical converter; the enumerators'
    ``to_atoms`` methods and :mod:`archimono.export` delegate to it.

    Args:
        tiling: The Archimedean tiling the assignment was generated for.
        supercell: ``(na, nb)`` tuple or 2×2 integer matrix supercell used
            during enumeration (matching :meth:`Tiling.graph`).
        assignment: Binary tuple of length *n* (0 = ``species[0]``,
            1 = ``species[1]``).  Atom ordering matches the node ordering of
            :meth:`~archimono.tilings.base.Tiling.graph`.
        species: Pair of ASE-valid element symbols ``(species_0, species_1)``.

    Returns:
        ASE Atoms object with correct cell, PBC ``[True, True, False]``,
        Cartesian positions, and chemical symbols assigned from *species*.

    Raises:
        ValueError: If ``len(assignment)`` does not match the number of atoms
            in the supercell, or if *assignment* contains values other than
            0 and 1.
    """
    atoms = _tiling_to_atoms(tiling, supercell)
    n = len(atoms)
    if len(assignment) != n:
        raise ValueError(
            f"assignment length {len(assignment)} does not match "
            f"supercell atom count {n}."
        )
    invalid = set(assignment) - {0, 1}
    if invalid:
        raise ValueError(
            f"assignment must be binary (0/1); got values {sorted(invalid)}."
        )
    symbols = [species[1] if a == 1 else species[0] for a in assignment]
    atoms.set_chemical_symbols(symbols)  # type: ignore[no-untyped-call]
    return atoms


def _get_symmetry_permutations(atoms: Atoms) -> list[tuple[int, ...]]:
    """Extract the space-group symmetry as site-index permutations via spglib.

    All atoms are assigned species number 1 (hydrogen) so that spglib returns
    the full crystallographic space group rather than a subgroup.

    For each symmetry operation (rotation R, translation t), the transformed
    fractional positions are ``new_pos[i] = R @ pos[i] + t`` (mod 1).  The
    permutation *perm* satisfies ``new_pos[i] ≈ pos[perm[i]]``, i.e.
    ``perm[i]`` is the site that site *i* maps to under the operation
    (push-forward / image convention).  Because the returned set of
    permutations is closed under inversion, canonical-form orbit minimization
    in :func:`~archimono.assignment.enumeration.canonical.canonical_form` is
    independent of this direction.

    References:
        Togo et al. (2018); see module docstring.
    """
    lattice = np.array(atoms.cell[:])
    positions = np.array(atoms.get_scaled_positions())  # type: ignore[no-untyped-call]
    numbers: list[int] = [1] * len(atoms)
    spglib_cell = (lattice.tolist(), positions.tolist(), numbers)

    symmetry = spglib.get_symmetry(spglib_cell)
    if symmetry is None:
        raise RuntimeError(
            "spglib.get_symmetry returned None.  "
            "Check that the geometry is physically reasonable."
        )

    n = len(atoms)
    perms: set[tuple[int, ...]] = set()

    for op_idx, (rot, trans) in enumerate(
        zip(symmetry["rotations"], symmetry["translations"])
    ):
        new_pos = (rot @ positions.T).T + trans
        new_pos = new_pos % 1.0

        perm = [-1] * n
        for i in range(n):
            for j in range(n):
                diff = new_pos[i] - positions[j]
                diff = diff - np.round(diff)
                if np.allclose(diff, 0, atol=1e-4):
                    perm[i] = j
                    break

        if -1 in perm:
            missing = [i for i, p in enumerate(perm) if p == -1]
            raise RuntimeError(
                f"spglib symmetry operation #{op_idx} did not map sites "
                f"{missing} to any site within atol=1e-4.  This indicates a "
                "geometry/symmetry inconsistency rather than a true symmetry."
            )
        if len(set(perm)) != n:
            raise RuntimeError(
                f"spglib symmetry operation #{op_idx} produced a non-injective "
                f"site map: {perm}.  Geometry or atol is inconsistent."
            )
        perms.add(tuple(perm))

    return list(perms)
