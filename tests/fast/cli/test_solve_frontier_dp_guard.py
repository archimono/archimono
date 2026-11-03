"""Fast tests for solve --solver frontier-dp's always-enumerate + memory guard."""

from __future__ import annotations

import json

import numpy as np
import pytest

from archimono.workflows import solve as solve_workflow


def test_json_ready_converts_ndarray_to_nested_lists() -> None:
    """The ndarray branch renders nested JSON integer lists, not a numpy repr."""
    array = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert solve_workflow._json_ready(array) == [[0, 1], [1, 0]]


def test_frontier_dp_guard_raises_when_estimate_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-limit estimate raises ValueError naming the limit and alternative."""
    monkeypatch.setattr(
        solve_workflow, "estimate_reconstruct_memory_bytes", lambda dp_cost, n: 10**15
    )
    config = solve_workflow.SolveConfig(
        tiling="hexagonal",
        supercell="2x2",
        solver="frontier-dp",
        memory_limit_gb=32.0,
    )

    with pytest.raises(ValueError) as excinfo:
        solve_workflow.run_solve(config)

    message = str(excinfo.value)
    assert "32" in message
    assert "frontier-dp-value" in message


def test_frontier_dp_default_solve_returns_one_witness() -> None:
    config = solve_workflow.SolveConfig(
        tiling="kagome",
        supercell="2x2",
        n_b=1,
        solver="frontier-dp",
    )

    result = solve_workflow.run_solve(config)
    assert result.labels is not None
    assert "all_labels" not in result.metadata


def test_frontier_dp_enumerate_all_returns_all_labels() -> None:
    """Explicit enumerate_all=True returns all_labels as uint8 ndarray."""
    config = solve_workflow.SolveConfig(
        tiling="kagome",
        supercell="2x2",
        n_b=1,
        solver="frontier-dp",
        enumerate_all=True,
    )

    result = solve_workflow.run_solve(config)
    all_labels = result.metadata["all_labels"]
    assert isinstance(all_labels, np.ndarray)
    assert all_labels.dtype == np.uint8
    assert all_labels.ndim == 2
    assert all_labels.shape[0] >= 1


def test_frontier_dp_json_emits_all_labels_as_nested_integer_arrays() -> None:
    config = solve_workflow.SolveConfig(
        tiling="kagome",
        supercell="2x2",
        n_b=1,
        solver="frontier-dp",
        enumerate_all=True,
    )
    result = solve_workflow.run_solve(config)

    text = solve_workflow.format_json(result)
    assert "array(" not in text
    payload = json.loads(text)
    all_labels = payload["metadata"]["all_labels"]
    assert isinstance(all_labels, list)
    assert all(isinstance(row, list) for row in all_labels)
    assert all(isinstance(v, int) for row in all_labels for v in row)


def test_frontier_dp_text_has_no_numpy_repr() -> None:
    config = solve_workflow.SolveConfig(
        tiling="kagome",
        supercell="2x2",
        n_b=1,
        solver="frontier-dp",
        enumerate_all=True,
    )
    result = solve_workflow.run_solve(config)

    text = solve_workflow.format_text(result)
    assert "array(" not in text
