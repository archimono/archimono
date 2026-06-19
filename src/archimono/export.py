"""Export enumerated binary assignments as structure files.

Provides :func:`assignment_to_atoms` to convert a binary assignment tuple to
an ASE :class:`~ase.Atoms` object, :func:`write_structure` to write a single
structure to disk, and :func:`export_structures` for batch export of all
enumerated inequivalent configurations.

Supported formats are anything ASE's ``ase.io.write`` accepts — common ones
include ``"vasp"`` (POSCAR), ``"cif"``, ``"xyz"``, and ``"extxyz"``.

Example (kagome 2×2, n_b=6 → 30 inequivalent structures):

.. code-block:: python

    from archimono.tilings import registry
    from archimono.assignment import OrderlyEnumerator
    from archimono.export import export_structures

    tiling = registry.get("kagome")
    assignments = OrderlyEnumerator().enumerate(tiling, n_b=6, supercell=(2, 2))
    paths = export_structures(
        tiling, (2, 2), assignments, outdir="out/kagome_2x2",
        species=("B", "N"), fmt="vasp",
    )
    len(paths)  # 30
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ase.io
from ase import Atoms

# assignment_to_atoms lives in the enumeration layer (next to tiling_to_atoms);
# re-exported here so ``archimono.export.assignment_to_atoms`` stays the public
# entry point for the export workflow.
from archimono.assignment.enumeration import assignment_to_atoms

if TYPE_CHECKING:
    from archimono.tilings.base import SupercellLike, Tiling

__all__ = ["assignment_to_atoms", "write_structure", "export_structures"]

# Map ASE format strings to conventional file extensions.
_FMT_TO_EXT: dict[str, str] = {
    "vasp": "vasp",
    "cif": "cif",
    "xyz": "xyz",
    "extxyz": "extxyz",
    "json": "json",
    "proteindatabank": "pdb",
    "aims": "in",
    "espresso-in": "pwi",
    "espresso-out": "pwo",
}


def write_structure(
    atoms: Atoms,
    path: str | Path,
    fmt: str | None = None,
) -> None:
    """Write an ASE Atoms object to a file.

    Format is inferred from the file extension when *fmt* is omitted.  Pass
    an explicit *fmt* string (``"vasp"``, ``"cif"``, ``"xyz"``, …) to
    override.

    Args:
        atoms: Structure to write.
        path: Destination file path.
        fmt: ASE format string.  ``None`` lets ASE infer from the extension.

    Raises:
        ase.io.formats.UnknownFileTypeError: If the format cannot be inferred
            from the extension and *fmt* is ``None``.
    """
    ase.io.write(str(path), atoms, format=fmt)


def export_structures(
    tiling: Tiling,
    supercell: SupercellLike,
    assignments: list[tuple[int, ...]],
    outdir: str | Path,
    *,
    species: tuple[str, str] = ("B", "N"),
    fmt: str = "vasp",
    prefix: str = "struct",
) -> list[Path]:
    """Export all assignments as structure files to a directory.

    Writes one file per assignment.  Files are named
    ``{prefix}_{i:04d}.{ext}`` where the extension is derived from *fmt*.

    Args:
        tiling: Archimedean tiling used for enumeration.
        supercell: ``(na, nb)`` tuple or 2×2 integer matrix supercell used
            during enumeration (matching :meth:`Tiling.graph`).
        assignments: List of binary assignment tuples as returned by
            :meth:`~archimono.assignment.OrderlyEnumerator.enumerate` or
            :meth:`~archimono.assignment.IcetEnumerator.enumerate`.
        outdir: Directory to write files into.  Created if absent.
        species: Element symbols ``(species_0, species_1)``.
        fmt: ASE format string (e.g. ``"vasp"``, ``"cif"``, ``"xyz"``).
        prefix: Filename prefix before the zero-padded index.

    Returns:
        List of :class:`~pathlib.Path` objects for the written files, in the
        same order as *assignments*.

    Raises:
        ValueError: If *assignments* is empty.

    Note:
        Files are written into *outdir* without clearing it first. Re-running
        with fewer assignments leaves higher-indexed files from a previous run
        in place, so use a fresh directory when the count may shrink.
    """
    if not assignments:
        raise ValueError("assignments must be non-empty.")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ext = _FMT_TO_EXT.get(fmt, fmt)

    paths: list[Path] = []
    for i, assignment in enumerate(assignments):
        atoms = assignment_to_atoms(tiling, supercell, assignment, species)
        dest = outdir / f"{prefix}_{i:04d}.{ext}"
        write_structure(atoms, dest, fmt=fmt)
        paths.append(dest)

    return paths
