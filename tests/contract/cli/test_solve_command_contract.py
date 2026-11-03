"""Contract tests for the ``archimono solve`` command."""

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
    environment["NUMBA_DISABLE_JIT"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "archimono.cli", *args],
        capture_output=True,
        encoding="utf-8",
        env=environment,
        timeout=120,
    )


def _cut_value(labels: Sequence[int]) -> int:
    """Recompute the periodic hexagonal cut value.

    Args:
        labels: Binary labels in periodic graph node order.

    Returns:
        Count of edges whose endpoints have different labels.
    """
    graph = registry.get("hexagonal").graph((2, 2))
    return sum(labels[u] != labels[v] for u, v in graph.edges())


def _text_fields(output: str) -> dict[str, str]:
    """Parse key/value fields from the text solve representation.

    Args:
        output: Text emitted by the solve command.

    Returns:
        Mapping from field names to their rendered values.
    """
    return {
        key: value.strip()
        for line in output.splitlines()
        for key, value in [line.split(":", maxsplit=1)]
    }


def _assert_error_result(result: subprocess.CompletedProcess[str]) -> None:
    """Assert a surfaced workflow error has no normal result or traceback.

    Args:
        result: CLI invocation expected to fail validation.
    """
    assert result.returncode == 1
    assert result.stderr.strip()
    assert "traceback" not in result.stderr.lower()
    assert result.stdout == ""


def test_bruteforce_solve_json_matches_the_periodic_graph_cut() -> None:
    """Bruteforce JSON labels reproduce all reported graph-derived scalars."""
    result = _run_cli(
        "solve",
        "hexagonal",
        "--supercell",
        "2x2",
        "--nb",
        "1",
        "--solver",
        "bruteforce",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    labels = payload["labels"]
    graph = registry.get("hexagonal").graph((2, 2))
    assert isinstance(labels, list)
    assert len(labels) == graph.number_of_nodes()
    assert set(labels) <= {0, 1}
    assert sum(labels) == 1
    assert _cut_value(labels) == payload["cut_value"]
    assert graph.number_of_edges() - payload["cut_value"] == payload["n_frustrated"]


def test_solve_text_uses_custom_species_symbols() -> None:
    result = _run_cli(
        "solve",
        "hexagonal",
        "--supercell",
        "2x2",
        "--nb",
        "1",
        "--solver",
        "bruteforce",
        "--species",
        "Li",
        "F",
    )

    assert result.returncode == 0, result.stderr
    labels = _text_fields(result.stdout)["labels"].split()
    assert len(labels) == 8
    assert set(labels) == {"Li", "F"}
    assert labels.count("F") == 1


def test_solve_text_and_json_agree_on_scalar_result_fields() -> None:
    arguments = (
        "solve",
        "hexagonal",
        "--supercell",
        "2x2",
        "--nb",
        "1",
        "--solver",
        "bruteforce",
    )
    text_result = _run_cli(*arguments, "--species", "Li", "F")
    json_result = _run_cli(*arguments, "--json")

    assert text_result.returncode == 0, text_result.stderr
    assert json_result.returncode == 0, json_result.stderr
    fields = _text_fields(text_result.stdout)
    payload = json.loads(json_result.stdout)
    assert fields["tiling"] == payload["tiling"]
    assert fields["supercell"] == "x".join(map(str, payload["supercell"]))
    assert fields["pbc"] == str(payload["pbc"]).lower()
    assert fields["solver"] == payload["solver"]
    assert int(fields["cut_value"]) == payload["cut_value"]
    assert int(fields["n_frustrated"]) == payload["n_frustrated"]


def test_frontier_dp_value_is_no_longer_a_solve_choice() -> None:
    result = _run_cli(
        "solve",
        "kagome",
        "--supercell",
        "1x1",
        "--nb",
        "1",
        "--solver",
        "frontier-dp-value",
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_frontier_dp_json_emits_all_labels_as_nested_integer_arrays() -> None:
    result = _run_cli(
        "solve",
        "kagome",
        "--supercell",
        "2x2",
        "--nb",
        "1",
        "--solver",
        "frontier-dp",
        "--all-optima",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    assert "array(" not in result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload["labels"], list)
    all_labels = payload["metadata"]["all_labels"]
    assert isinstance(all_labels, list)
    assert len(all_labels) >= 1
    assert all(isinstance(row, list) for row in all_labels)
    assert all(isinstance(v, int) for row in all_labels for v in row)


def test_hadlock_solves_a_planar_unconstrained_square_graph() -> None:
    """Hadlock accepts the non-periodic square graph without composition."""
    result = _run_cli(
        "solve",
        "square",
        "--supercell",
        "2x2",
        "--solver",
        "hadlock",
        "--no-pbc",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["pbc"] is False
    assert payload["solver"] == "hadlock"


def test_hadlock_rejects_a_composition_constraint() -> None:
    result = _run_cli(
        "solve",
        "square",
        "--supercell",
        "2x2",
        "--solver",
        "hadlock",
        "--no-pbc",
        "--nb",
        "2",
    )

    _assert_error_result(result)
    assert "composition" in result.stderr.lower()


def test_hadlock_rejects_the_default_periodic_graph() -> None:
    """Hadlock reports that the default periodic graph is not planar."""
    result = _run_cli("solve", "square", "--supercell", "2x2", "--solver", "hadlock")

    _assert_error_result(result)
    assert "planar" in result.stderr.lower()


def test_solve_unknown_tiling_returns_a_diagnostic() -> None:
    """Unknown tiling keys are surfaced as workflow errors."""
    result = _run_cli("solve", "does-not-exist", "--json")

    _assert_error_result(result)
    assert "does-not-exist" in result.stderr


def test_solve_malformed_supercell_returns_a_diagnostic() -> None:
    """Malformed supercell strings are surfaced as workflow errors."""
    result = _run_cli("solve", "kagome", "--supercell", "not-a-supercell")

    _assert_error_result(result)
    assert "supercell" in result.stderr.lower()


def test_solve_wrong_species_arity_returns_a_diagnostic() -> None:
    """The solve workflow rejects species lists that are not binary."""
    result = _run_cli("solve", "kagome", "--species", "A", "B", "C")

    _assert_error_result(result)
    assert "exactly two" in result.stderr.lower()


def test_solve_out_of_range_composition_returns_a_diagnostic() -> None:
    """Composition above the graph node count is surfaced as an error."""
    result = _run_cli(
        "solve",
        "kagome",
        "--supercell",
        "2x2",
        "--nb",
        "13",
        "--json",
    )

    _assert_error_result(result)
    assert "n_b" in result.stderr
    assert "12" in result.stderr
