"""Contract smoke test for scripts/export_structures.py."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest


@pytest.mark.contract
def test_export_structures_smoke(tmp_path: pathlib.Path) -> None:
    """Run export_structures.py end-to-end on kagome 2x2 and check output.

    Kagome 2x2 with n_b=6 has 30 symmetry-inequivalent configurations (the
    canonical reference case), so 30 files must be written.
    """
    script = (
        pathlib.Path(__file__).parents[3] / "scripts" / "export_structures.py"
    )
    outdir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--tiling",
            "kagome",
            "--supercell",
            "2",
            "2",
            "--n-b",
            "6",
            "--outdir",
            str(outdir),
            "--format",
            "xyz",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"export_structures.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    files = sorted(outdir.glob("*.xyz"))
    assert len(files) == 30, f"expected 30 .xyz files, got {len(files)}"
