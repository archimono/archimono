"""Fast parser-to-workflow adapter tests for ``archimono solve``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

import archimono
import archimono.workflows
from archimono.cli import __main__

pytestmark = pytest.mark.fast


@dataclass
class _SolveConfig:
    """Complete stand-in for the typed solve-workflow configuration."""

    tiling: str
    supercell: str
    n_b: int | None
    solver: str
    species: object
    pbc: bool
    emit_json: bool
    memory_limit_gb: float
    enumerate_all: bool
    max_optima: int


def _install_workflow(
    monkeypatch: pytest.MonkeyPatch,
    solve: object,
) -> list[_SolveConfig]:
    """Install a capture-only solve workflow at the command's lazy seam."""
    captured: list[_SolveConfig] = []
    workflow = ModuleType("archimono.workflows.solve")
    workflow.SolveConfig = _SolveConfig

    def capture(config: _SolveConfig) -> object:
        captured.append(config)
        return solve(config)  # type: ignore[operator]

    workflow.run_solve = capture
    workflow.format_text = lambda result: "text result"
    workflow.format_json = lambda result: "json result"
    monkeypatch.setitem(sys.modules, "archimono.workflows.solve", workflow)
    monkeypatch.setattr(
        archimono.workflows, "solve", workflow, raising=False
    )
    return captured


def test_solve_defaults_are_forwarded_to_the_solve_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflow(monkeypatch, lambda config: object())

    assert __main__.main(["solve", "square"]) == 0
    assert captured == [
        _SolveConfig(
            "square", "1x1", None, "bruteforce", ("A", "B"), True, False, 32.0,
            False, 10_000,
        )
    ]


def test_solve_explicit_options_are_forwarded_to_the_solve_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflow(monkeypatch, lambda config: object())

    assert __main__.main(
        [
            "solve",
            "kagome",
            "--supercell",
            "2,2",
            "--nb",
            "6",
            "--solver",
            "greedy",
            "--species",
            "Fe",
            "Ni",
            "--no-pbc",
            "--json",
            "--memory-limit-gb",
            "8",
        ]
    ) == 0
    config = captured[0]
    assert config.tiling == "kagome"
    assert config.supercell == "2,2"
    assert config.n_b == 6
    assert config.solver == "greedy"
    assert tuple(config.species) == ("Fe", "Ni")
    assert config.pbc is False
    assert config.emit_json is True
    assert config.memory_limit_gb == 8.0
    assert config.enumerate_all is False
    assert config.max_optima == 10_000


def test_solve_workflow_value_error_is_written_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Workflow validation failures are process errors without normal output."""
    def fail(config: _SolveConfig) -> object:
        del config
        raise ValueError("invalid composition")

    _install_workflow(monkeypatch, fail)

    assert __main__.main(["solve", "square"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid composition" in captured.err


def test_solve_unknown_tiling_error_is_written_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(config: _SolveConfig) -> object:
        del config
        raise KeyError("missing-tiling")

    _install_workflow(monkeypatch, fail)

    assert __main__.main(["solve", "missing-tiling"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing-tiling" in captured.err


def test_solver_choices_match_canonical_registry() -> None:
    """The hand-listed --solver choices stay in sync with SOLVERS_BY_NAME."""
    from archimono.assignment import SOLVERS_BY_NAME
    from archimono.cli.solve import _SOLVERS

    assert _SOLVERS == tuple(SOLVERS_BY_NAME)
