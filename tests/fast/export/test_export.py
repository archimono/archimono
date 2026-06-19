"""Fast unit tests for archimono.export."""

from __future__ import annotations

import pathlib

import ase.io
import numpy as np
import pytest

from archimono.assignment import IcetEnumerator, OrderlyEnumerator
from archimono.export import (
    assignment_to_atoms,
    export_structures,
    write_structure,
)
from archimono.tilings import registry

_SUPERCELL = (2, 2)
_N_B = 6  # kagome 2x2 has n=12; 6 label-1 sites


def _kagome_assignments() -> list[tuple[int, ...]]:
    tiling = registry.get("kagome")
    return OrderlyEnumerator().enumerate(tiling, n_b=_N_B, supercell=_SUPERCELL)


class TestAssignmentToAtoms:
    """assignment_to_atoms maps labels to species and validates length."""

    def test_symbols_count_and_set(self) -> None:
        """Output has n atoms, only the two species, n_b of species[1]."""
        tiling = registry.get("kagome")
        assignment = _kagome_assignments()[0]

        atoms = assignment_to_atoms(
            tiling, _SUPERCELL, assignment, species=("B", "N")
        )
        symbols = atoms.get_chemical_symbols()

        assert len(symbols) == 12
        assert set(symbols) <= {"B", "N"}
        assert symbols.count("N") == _N_B  # species[1] count == n_b
        assert sum(assignment) == _N_B

    def test_species_are_applied(self) -> None:
        """A different species pair is honoured."""
        tiling = registry.get("kagome")
        assignment = _kagome_assignments()[0]

        atoms = assignment_to_atoms(
            tiling, _SUPERCELL, assignment, species=("Mo", "S")
        )

        assert set(atoms.get_chemical_symbols()) <= {"Mo", "S"}

    def test_length_mismatch_raises(self) -> None:
        """A wrong-length assignment raises ValueError."""
        tiling = registry.get("kagome")
        with pytest.raises(ValueError, match="does not match"):
            assignment_to_atoms(tiling, _SUPERCELL, (0, 1, 0))

    def test_non_binary_value_raises(self) -> None:
        """An assignment with values outside {0, 1} raises ValueError."""
        tiling = registry.get("kagome")
        bad = (2,) + (0,) * 11  # right length, illegal value
        with pytest.raises(ValueError, match="must be binary"):
            assignment_to_atoms(tiling, _SUPERCELL, bad)


class TestExportStructures:
    """export_structures writes one file per assignment."""

    def test_file_count_and_paths(self, tmp_path: pathlib.Path) -> None:
        """Writes len(assignments) files with the format's extension."""
        tiling = registry.get("kagome")
        subset = _kagome_assignments()[:3]

        paths = export_structures(
            tiling, _SUPERCELL, subset, outdir=tmp_path, fmt="vasp"
        )

        assert len(paths) == 3
        assert all(p.exists() for p in paths)
        assert all(p.suffix == ".vasp" for p in paths)
        assert sorted(p.name for p in paths) == [
            "struct_0000.vasp",
            "struct_0001.vasp",
            "struct_0002.vasp",
        ]

    def test_empty_assignments_raises(self, tmp_path: pathlib.Path) -> None:
        """An empty assignment list raises ValueError."""
        tiling = registry.get("kagome")
        with pytest.raises(ValueError, match="non-empty"):
            export_structures(tiling, _SUPERCELL, [], outdir=tmp_path)

    def test_creates_missing_outdir(self, tmp_path: pathlib.Path) -> None:
        """A non-existent output directory is created."""
        tiling = registry.get("kagome")
        outdir = tmp_path / "nested" / "out"

        paths = export_structures(
            tiling, _SUPERCELL, _kagome_assignments()[:1], outdir=outdir
        )

        assert outdir.is_dir()
        assert len(paths) == 1


class TestEnumeratorToAtomsDelegation:
    """Both enumerators' to_atoms delegate to the validated export function."""

    @pytest.mark.parametrize(
        "enumerator", [OrderlyEnumerator, IcetEnumerator], ids=["orderly", "icet"]
    )
    def test_matches_assignment_to_atoms(
        self, enumerator: type[OrderlyEnumerator] | type[IcetEnumerator]
    ) -> None:
        """to_atoms produces the same structure as assignment_to_atoms.

        ``to_atoms`` delegates to ``export.assignment_to_atoms`` and does not
        touch ``icet.tools``, so the icet variant works without icet installed.
        """
        tiling = registry.get("kagome")
        assignment = _kagome_assignments()[0]

        via_method = enumerator.to_atoms(tiling, _SUPERCELL, assignment)
        via_func = assignment_to_atoms(tiling, _SUPERCELL, assignment)

        assert (
            via_method.get_chemical_symbols() == via_func.get_chemical_symbols()
        )

    @pytest.mark.parametrize(
        "enumerator", [OrderlyEnumerator, IcetEnumerator], ids=["orderly", "icet"]
    )
    def test_length_validation_delegated(
        self, enumerator: type[OrderlyEnumerator] | type[IcetEnumerator]
    ) -> None:
        """to_atoms raises ValueError on a wrong-length assignment."""
        tiling = registry.get("kagome")
        with pytest.raises(ValueError, match="does not match"):
            enumerator.to_atoms(tiling, _SUPERCELL, (0, 1, 0))


class TestWriteStructure:
    """write_structure round-trips through ASE."""

    @pytest.mark.parametrize("fmt", ["extxyz", "vasp"], ids=["extxyz", "vasp"])
    def test_roundtrip_preserves_atoms_and_cell(
        self, tmp_path: pathlib.Path, fmt: str
    ) -> None:
        """A written structure reads back with same atom count and cell.

        Uses cell-preserving formats (the formats DFT users export to); plain
        ``xyz`` is intentionally excluded as it drops the cell.
        """
        tiling = registry.get("kagome")
        atoms = assignment_to_atoms(
            tiling, _SUPERCELL, _kagome_assignments()[0]
        )
        dest = tmp_path / f"s.{fmt}"

        write_structure(atoms, dest, fmt=fmt)
        back = ase.io.read(str(dest), format=fmt)

        assert len(back) == len(atoms)
        assert np.allclose(back.cell.array, atoms.cell.array, atol=1e-4)
