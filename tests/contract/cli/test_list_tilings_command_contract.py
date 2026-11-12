"""Contract tests for the ``archimono list-tilings`` command."""

from __future__ import annotations

import os
import subprocess
import sys

from archimono.tilings import registry


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


def test_list_tilings_prints_registered_keys_and_vertex_counts() -> None:
    result = _run_cli("list-tilings")

    expected_lines = [
        f"{key} {registry.get(key).n_vertices}" for key in registry.available()
    ]
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected_lines
    assert result.stderr == ""


def test_list_tilings_emits_utf8_when_parent_stdio_is_cp1252() -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"

    result = subprocess.run(
        [sys.executable, "-m", "archimono.cli", "list-tilings"],
        capture_output=True,
        env=env,
        timeout=120,
    )

    expected_lines = [
        f"{key} {registry.get(key).n_vertices}" for key in registry.available()
    ]
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    output = result.stdout.decode("utf-8")
    assert output.splitlines() == expected_lines
    assert result.stderr == b""
