"""Export symmetry-inequivalent structures to structure files.

Enumerates all inequivalent binary assignments for a given tiling and
supercell, then writes each structure to a directory in any format supported
by ASE (POSCAR, CIF, XYZ, …).

Example
-------
Export kagome (2×2, n_b=6) structures as VASP POSCAR files::

    python scripts/export_structures.py \\
        --tiling kagome \\
        --supercell 2 2 \\
        --n-b 6 \\
        --outdir out/kagome_2x2 \\
        --format vasp \\
        --species B N

Export hexagonal (1×1) structures as CIF::

    python scripts/export_structures.py \\
        --tiling hexagonal \\
        --supercell 1 1 \\
        --n-b 1 \\
        --outdir out/hexagonal_1x1 \\
        --format cif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from archimono.assignment import enumeration
from archimono.export import export_structures
from archimono.tilings import registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate symmetry-inequivalent structures for an Archimedean tiling "
            "and export them as structure files."
        )
    )
    parser.add_argument(
        "--tiling",
        required=True,
        metavar="KEY",
        help="Registry key of the tiling (e.g. kagome, hexagonal, triangular).",
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs="+",
        default=[1, 1],
        metavar="N",
        help=(
            "Supercell as two ints (na nb) for a diagonal cell, or four ints "
            "(a b c d) for the 2x2 integer matrix [[a, b], [c, d]] (sheared "
            "cell, e.g. needed for the 4.8.8 tiling at n=24).  Default: 1 1."
        ),
    )
    parser.add_argument(
        "--n-b",
        type=int,
        required=True,
        metavar="N",
        help="Number of species[1] (label-1) atoms.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory.  Created if absent.",
    )
    parser.add_argument(
        "--format",
        default="vasp",
        metavar="FMT",
        help="ASE format string (vasp, cif, xyz, extxyz, …).  Default: vasp.",
    )
    parser.add_argument(
        "--species",
        nargs=2,
        default=["B", "N"],
        metavar=("A", "B"),
        help="Element symbols for label-0 and label-1 sites.  Default: B N.",
    )
    parser.add_argument(
        "--prefix",
        default="struct",
        metavar="STR",
        help="Filename prefix before the zero-padded index.  Default: struct.",
    )
    parser.add_argument(
        "--enumerator",
        choices=["orderly", "icet"],
        default="orderly",
        help="Enumeration backend.  Default: orderly.",
    )
    parser.add_argument(
        "--min-cut",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Keep only configurations with MAX-CUT value >= K.  "
            "Default: 0 (all configurations)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the export script."""
    args = _parse_args()

    tiling = registry.get(args.tiling)
    sc_vals = args.supercell
    if len(sc_vals) == 2:
        supercell: tuple[int, int] | np.ndarray = (sc_vals[0], sc_vals[1])
        sc_label = f"{sc_vals[0]}×{sc_vals[1]}"
    elif len(sc_vals) == 4:
        supercell = np.array(
            [[sc_vals[0], sc_vals[1]], [sc_vals[2], sc_vals[3]]], dtype=np.intp
        )
        sc_label = f"matrix {supercell.tolist()}"
    else:
        raise SystemExit(
            f"--supercell expects 2 ints (na nb) or 4 ints (a b c d for a 2x2 "
            f"matrix); got {len(sc_vals)}: {sc_vals}."
        )
    species: tuple[str, str] = (args.species[0], args.species[1])

    print(
        f"Tiling   : {args.tiling}  (n_vertices={tiling.n_vertices},"
        f"  coordination={tiling.coordination})",
        flush=True,
    )
    print(f"Supercell: {sc_label}", flush=True)
    print(f"n_b      : {args.n_b}", flush=True)
    print(f"Species  : {species[0]} (label 0)  /  {species[1]} (label 1)", flush=True)
    print(f"Format   : {args.format}", flush=True)
    print(f"Out dir  : {args.outdir}", flush=True)
    print(f"Enumerator: {args.enumerator}", flush=True)
    print("", flush=True)

    print("Enumerating …", flush=True)
    assignments = enumeration.get(args.enumerator).enumerate(
        tiling,
        n_b=args.n_b,
        supercell=supercell,
        min_cut=args.min_cut,
    )

    print(f"Found {len(assignments)} inequivalent structure(s).", flush=True)

    print("Writing files …", flush=True)
    paths = export_structures(
        tiling,
        supercell,
        assignments,
        outdir=args.outdir,
        species=species,
        fmt=args.format,
        prefix=args.prefix,
    )

    for p in paths:
        print(f"  {p}", flush=True)

    print(f"\nDone. {len(paths)} file(s) written to {args.outdir}.", flush=True)


if __name__ == "__main__":
    main()
