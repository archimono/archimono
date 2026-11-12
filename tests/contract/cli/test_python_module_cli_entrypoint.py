"""Contract coverage for the Python-module CLI entry point."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from archimono.tilings import registry


def _run_module(
    cwd: pathlib.Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """Run the package CLI module with UTF-8 subprocess streams.

    Args:
        cwd: Working directory for the subprocess.
        *args: Arguments after ``python -m archimono.cli``.

    Returns:
        Completed CLI process result.
    """
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "archimono.cli", *args],
        capture_output=True,
        encoding="utf-8",
        cwd=cwd,
        env=environment,
        timeout=120,
    )


def test_module_entrypoint_lists_registered_tilings_and_vertex_counts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poisoned_cli = tmp_path / "poisoned" / "archimono" / "cli"
    poisoned_cli.mkdir(parents=True)
    (poisoned_cli.parent / "__init__.py").write_text("", encoding="utf-8")
    (poisoned_cli / "__init__.py").write_text("", encoding="utf-8")
    (poisoned_cli / "__main__.py").write_text(
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(poisoned_cli.parents[1]))

    result = _run_module(tmp_path, "list-tilings")

    rows = [tuple(line.split(maxsplit=1)) for line in result.stdout.splitlines()]
    expected = [
        (key, str(registry.get(key).n_vertices)) for key in registry.available()
    ]

    assert result.returncode == 0, result.stderr
    assert rows == expected
    assert result.stderr == ""
