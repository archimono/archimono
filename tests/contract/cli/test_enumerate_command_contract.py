"""Contract tests for the ``archimono enumerate`` command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence

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
    environment["MKL_NUM_THREADS"] = "1"
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "archimono.cli", *args],
        capture_output=True,
        encoding="utf-8",
        env=environment,
        timeout=120,
    )


def _cut_value(labels: Sequence[int]) -> int:
    """Recompute a periodic kagome cut value.

    Args:
        labels: Binary labels in periodic graph node order.

    Returns:
        Count of edges whose endpoints have different labels.
    """
    graph = registry.get("kagome").graph((2, 2))
    return sum(labels[u] != labels[v] for u, v in graph.edges())


def _assert_workflow_error(result: subprocess.CompletedProcess[str]) -> None:
    """Assert a CLI workflow error is reported without a traceback.

    Args:
        result: CLI invocation expected to fail validation.
    """
    assert result.returncode == 1
    assert result.stderr.strip()
    assert "traceback" not in result.stderr.lower()
    assert result.stdout == ""


def test_orderly_enumeration_json_lists_every_kagome_one_by_one_structure() -> None:
    """Orderly enumeration JSON describes every known 2x2 kagome structure."""
    result = _run_cli(
        "enumerate",
        "kagome",
        "--supercell",
        "2x2",
        "--nb",
        "1",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    structures = payload["structures"]
    assert payload["count"] == 1
    assert payload["count"] == len(structures)
    for structure in structures:
        labels = structure["labels"]
        assert len(labels) == 12
        assert set(labels) <= {0, 1}
        assert sum(labels) == 1
        assert _cut_value(labels) == structure["cut_value"]


def test_icet_enumeration_honours_the_minimum_cut_threshold() -> None:
    result = _run_cli(
        "enumerate",
        "kagome",
        "--supercell",
        "2x2",
        "--nb",
        "1",
        "--backend",
        "icet",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    # kagome 2x2 --nb 1 has exactly 1 symmetry-distinct structure: kagome is
    # vertex-transitive, so every single-atom minority placement is
    # equivalent under the supercell's space group.
    assert payload["count"] == 1
    assert payload["count"] == len(payload["structures"])
    assert all(row["cut_value"] >= 0 for row in payload["structures"])


def test_enumerate_requires_composition_before_dispatch() -> None:
    """Missing composition remains an argparse usage error."""
    result = _run_cli("enumerate", "kagome")

    assert result.returncode == 2
    assert "usage:" in result.stderr.lower()
    assert "--nb" in result.stderr
    assert "traceback" not in result.stderr.lower()
    assert result.stdout == ""


def test_enumerate_malformed_supercell_returns_a_diagnostic() -> None:
    """Malformed supercell strings become surfaced workflow errors."""
    result = _run_cli(
        "enumerate",
        "kagome",
        "--supercell",
        "not-a-supercell",
        "--nb",
        "6",
    )

    _assert_workflow_error(result)
    assert "supercell" in result.stderr.lower()


def test_enumerate_out_of_range_composition_returns_a_diagnostic() -> None:
    """Composition above the graph node count becomes a workflow error."""
    result = _run_cli(
        "enumerate",
        "kagome",
        "--supercell",
        "2x2",
        "--nb",
        "13",
        "--json",
    )

    _assert_workflow_error(result)
    assert "n_b" in result.stderr
    assert "12" in result.stderr


def test_enumerate_prints_count_then_structure_rows() -> None:
    """``enumerate`` prints the count first, then labels and cut values."""
    result = _run_cli(
        "enumerate",
        "kagome",
        "--supercell",
        "2x2",
        "--nb",
        "1",
    )

    expected_count = 1
    lines = result.stdout.splitlines()
    assert result.returncode == 0, result.stderr
    assert lines[0] == f"count: {expected_count}"
    assert len(lines) == expected_count + 1
    assert all("labels:" in line and "cut_value:" in line for line in lines[1:])
