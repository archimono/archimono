"""Contract smoke tests for the benchmark CLI command.

Runs two subcommands with minimal arguments and verifies their output
artifacts exist.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest


@pytest.mark.contract
def test_sample_runs_smoke(tmp_path: pathlib.Path) -> None:
    """Run benchmark sample-runs on hexagonal n=12 and verify DB exists.

    Invokes ``python -m archimono.cli benchmark sample-runs --db {tmp_path}/test.db
    --tilings hexagonal --target-sizes 12 --n-runs 2`` and checks that
    the SQLite database file exists.
    """
    db_path = tmp_path / "test.db"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "archimono.cli",
            "benchmark",
            "sample-runs",
            "--db",
            str(db_path),
            "--tilings",
            "hexagonal",
            "--target-sizes",
            "12",
            "--n-runs",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"benchmark.py sample-runs exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert db_path.exists(), (
        f"DB not found at {db_path}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
