"""Contract tests for the ``archimono verify`` command."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI module with UTF-8 subprocess streams.

    Args:
        *args: Arguments after ``python -m archimono.cli``.

    Returns:
        Completed CLI process result.
    """
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "archimono.cli", *args],
        capture_output=True,
        encoding="utf-8",
        env=environment,
        timeout=120,
    )


def test_geometry_verification_writes_reference_image(
    tmp_path: pathlib.Path,
) -> None:
    """Geometry verification writes the overview image and per-tiling PNGs."""
    output_dir = tmp_path / "verification"
    result = _run_cli(
        "verify",
        "--mode",
        "geometry",
        "--out-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    overview = output_dir / "overview_geometry.png"
    assert overview.is_file() and overview.stat().st_size > 0
