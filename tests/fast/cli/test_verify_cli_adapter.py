"""Fast parser-to-workflow adapter tests for ``archimono verify``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import archimono
from archimono.cli import __main__, verify

pytestmark = pytest.mark.fast


@dataclass
class _VerificationConfig:
    """Complete stand-in for verification workflow configuration."""

    mode: str
    out_dir: Path


def _install_workflow(
    monkeypatch: pytest.MonkeyPatch, ok: bool
) -> list[_VerificationConfig]:
    """Install a capture-only verification workflow at the command seam."""
    captured: list[_VerificationConfig] = []
    workflow = ModuleType("archimono.verification")
    workflow.VerificationConfig = _VerificationConfig

    def run(config: _VerificationConfig) -> SimpleNamespace:
        captured.append(config)
        return SimpleNamespace(ok=ok)

    workflow.run_verification = run
    monkeypatch.setitem(sys.modules, "archimono.verification", workflow)
    monkeypatch.setattr(archimono, "verification", workflow, raising=False)
    return captured


def test_verify_default_output_directory_is_repository_tmp_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflow(monkeypatch, ok=True)

    assert __main__.main(["verify"]) == 0
    assert captured[0].out_dir == verify._PROJECT_ROOT / "tmp" / "verify"


def test_verify_failed_workflow_result_returns_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_workflow(monkeypatch, ok=False)

    assert __main__.main(["verify"]) == 1
