"""Fast parser-to-workflow adapter tests for ``archimono enumerate``."""

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
class _EnumerationConfig:
    """Complete stand-in for the typed enumeration-workflow configuration."""

    tiling: str
    n_b: int
    supercell: str
    min_cut: int
    backend: str
    emit_json: bool


def _install_workflow(
    monkeypatch: pytest.MonkeyPatch,
    enumerate_structures: object,
) -> list[_EnumerationConfig]:
    """Install a capture-only enumeration workflow at the command seam."""
    captured: list[_EnumerationConfig] = []
    workflow = ModuleType("archimono.workflows.enumerate")
    workflow.EnumerationConfig = _EnumerationConfig

    def capture(config: _EnumerationConfig) -> object:
        captured.append(config)
        return enumerate_structures(config)  # type: ignore[operator]

    workflow.enumerate_structures = capture
    workflow.format_text = lambda result: "text result"
    workflow.format_json = lambda result: "json result"
    monkeypatch.setitem(sys.modules, "archimono.workflows.enumerate", workflow)
    monkeypatch.setattr(
        archimono.workflows, "enumerate", workflow, raising=False
    )
    return captured


def test_enumerate_defaults_are_forwarded_to_the_enumeration_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflow(monkeypatch, lambda config: object())

    assert __main__.main(["enumerate", "square", "--nb", "2"]) == 0
    assert captured == [
        _EnumerationConfig("square", 2, "1x1", 0, "orderly", False)
    ]


def test_enumerate_explicit_backend_and_minimum_cut_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflow(monkeypatch, lambda config: object())

    assert __main__.main(
        [
            "enumerate",
            "kagome",
            "--nb",
            "6",
            "--supercell",
            "2,2",
            "--backend",
            "icet",
            "--min-cut",
            "14",
            "--json",
        ]
    ) == 0
    assert captured == [
        _EnumerationConfig("kagome", 6, "2,2", 14, "icet", True)
    ]


def test_enumerate_workflow_value_error_is_written_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(config: _EnumerationConfig) -> object:
        del config
        raise ValueError("invalid supercell")

    _install_workflow(monkeypatch, fail)

    assert __main__.main(["enumerate", "square", "--nb", "2"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid supercell" in captured.err


def test_enumerate_unknown_tiling_error_is_written_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(config: _EnumerationConfig) -> object:
        del config
        raise KeyError("missing-tiling")

    _install_workflow(monkeypatch, fail)

    assert __main__.main(["enumerate", "missing-tiling", "--nb", "2"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing-tiling" in captured.err
