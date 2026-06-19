"""Contract smoke tests for scripts/benchmark.py.

Runs two subcommands with minimal arguments and verifies their output
artifacts exist.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest


@pytest.mark.contract
def test_frontier_dp_smoke(tmp_path: pathlib.Path) -> None:
    """Run benchmark.py frontier-dp on hexagonal n=12 and verify markdown output.

    Invokes ``scripts/benchmark.py frontier-dp --tilings hexagonal
    --target-sizes 12 --md-out {tmp_path}/bench.md`` and checks that the
    markdown file exists.

    Args:
        tmp_path: Pytest-provided temporary directory for markdown output.
    """
    script = pathlib.Path(__file__).parents[3] / "scripts" / "benchmark.py"
    md_out = tmp_path / "bench.md"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "frontier-dp",
            "--tilings",
            "hexagonal",
            "--target-sizes",
            "12",
            "--md-out",
            str(md_out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"benchmark.py frontier-dp exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert md_out.exists(), (
        f"Markdown output not found at {md_out}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.contract
def test_sample_runs_smoke(tmp_path: pathlib.Path) -> None:
    """Run benchmark.py sample-runs on hexagonal n=12 and verify DB exists.

    Invokes ``scripts/benchmark.py sample-runs --db {tmp_path}/test.db
    --tilings hexagonal --target-sizes 12 --n-runs 2`` and checks that
    the SQLite database file exists.

    Args:
        tmp_path: Pytest-provided temporary directory for the DB file.
    """
    script = pathlib.Path(__file__).parents[3] / "scripts" / "benchmark.py"
    db_path = tmp_path / "test.db"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
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
