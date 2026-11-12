"""Contract tests for certify.py provenance fields (audit M1).

M1: heuristic annealing on a frustrated tiling must be ``status="heuristic"``,
not ``"certified"`` (only the exact 2-colouring on bipartite tilings is
certified).
"""

from __future__ import annotations

import pathlib

import pytest

from archimono.assignment import AnnealingSolver
from archimono.tilings import registry
from archimono.workflows import certify as certification_workflows


def test_annealing_frustrated_is_heuristic(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frustrated tiling solved by annealing is labelled heuristic, not certified."""
    monkeypatch.setattr(
        certification_workflows,
        "_ANNEALING_SOLVER",
        AnnealingSolver(n_restarts=1, seed=42),
    )
    tiling = registry.get("triangular")
    record = certification_workflows._annealing_case(
        tiling=tiling,
        tiling_key="triangular",
        tiling_display="3⁶ (triangular)",
        target_n=12,
        out_dir=tmp_path,
    )
    assert record.solver == "annealing"
    assert record.status == "heuristic"
    assert record.metadata["is_exact"] is False


def test_annealing_bipartite_is_certified(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bipartite tiling solved by exact 2-colouring stays certified."""
    monkeypatch.setattr(
        certification_workflows,
        "_ANNEALING_SOLVER",
        AnnealingSolver(n_restarts=1, seed=42),
    )
    tiling = registry.get("hexagonal")
    record = certification_workflows._annealing_case(
        tiling=tiling,
        tiling_key="hexagonal",
        tiling_display="6³ (hexagonal)",
        target_n=12,
        out_dir=tmp_path,
    )
    assert record.solver == "bipartite_exact"
    assert record.status == "certified"
