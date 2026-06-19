"""Fast tests for scripts/verify.py main exit-code handling."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import types

import pytest


def _load_verify_module() -> types.ModuleType:
    """Load scripts/verify.py without requiring scripts to be a package."""
    script = pathlib.Path(__file__).parents[3] / "scripts" / "verify.py"
    spec = importlib.util.spec_from_file_location("verify_script", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_returns_nonzero_when_geometry_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A failed geometry runner makes main return a failing exit code."""
    verify = _load_verify_module()
    monkeypatch.setattr(
        verify,
        "_parse_args",
        lambda: argparse.Namespace(mode="geometry", out_dir=tmp_path),
    )
    monkeypatch.setattr(verify, "run_geometry", lambda out_dir: False)

    assert verify.main() == 1


def test_main_returns_nonzero_when_exact_solver_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A failed exact-solver runner makes main return a failing exit code."""
    verify = _load_verify_module()
    monkeypatch.setattr(
        verify,
        "_parse_args",
        lambda: argparse.Namespace(mode="exact-solver", out_dir=tmp_path),
    )
    monkeypatch.setattr(verify, "run_exact_solver", lambda out_dir: False)

    assert verify.main() == 1


def test_main_returns_zero_when_selected_checks_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Passing selected verification modes keep the process exit code zero."""
    verify = _load_verify_module()
    monkeypatch.setattr(
        verify,
        "_parse_args",
        lambda: argparse.Namespace(mode="all", out_dir=tmp_path),
    )
    monkeypatch.setattr(verify, "run_geometry", lambda out_dir: True)
    monkeypatch.setattr(verify, "run_exact_solver", lambda out_dir: True)
    monkeypatch.setattr(verify, "run_visualize", lambda out_dir: None)

    assert verify.main() == 0
