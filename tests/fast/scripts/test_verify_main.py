"""Fast tests for the verify command adapter."""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
import types

import pytest

from archimono.cli import verify

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_VERIFY_MODULE = _PROJECT_ROOT / "src" / "archimono" / "cli" / "verify.py"


def test_verify_cli_exposes_only_command_seam() -> None:
    public_names = sorted(
        name for name in vars(verify) if not name.startswith("_")
    )

    assert public_names == ["configure_parser", "run"]


def test_verify_cli_has_no_top_level_workflow_imports() -> None:
    banned_modules = {
        "matplotlib",
        "networkx",
        "numpy",
        "sqlite3",
        "archimono.assignment",
        "archimono.tilings.registry",
        "archimono.verification",
    }
    tree = ast.parse(_VERIFY_MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "TYPE_CHECKING":
                continue
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
            imported.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if node.module == "archimono"
            )

    assert not (imported & banned_modules)


def test_run_delegates_to_verification_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The CLI adapter converts parsed args into a workflow config."""
    captured_config: list[object] = []
    fake_module = types.ModuleType("archimono.verification")

    class FakeConfig:
        """Minimal stand-in for the workflow config dataclass."""

        def __init__(self, *, mode: str, out_dir: pathlib.Path) -> None:
            self.mode = mode
            self.out_dir = out_dir

    class FakeResult:
        """Minimal stand-in for the workflow result dataclass."""

        ok = False

    def fake_run_verification(config: object) -> FakeResult:
        captured_config.append(config)
        return FakeResult()

    fake_module.VerificationConfig = FakeConfig
    fake_module.run_verification = fake_run_verification
    monkeypatch.setitem(sys.modules, "archimono.verification", fake_module)
    monkeypatch.setattr(
        verify,
        "run_geometry",
        lambda _out_dir: pytest.fail("CLI run must delegate to workflow module"),
        raising=False,
    )

    args = argparse.Namespace(mode="geometry", out_dir=tmp_path)

    assert verify.run(args) == 1
    assert len(captured_config) == 1
    assert captured_config[0].mode == "geometry"
    assert captured_config[0].out_dir == tmp_path


def test_run_returns_zero_when_workflow_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    fake_module = types.ModuleType("archimono.verification")

    class FakeConfig:
        """Minimal stand-in for the workflow config dataclass."""

        def __init__(self, *, mode: str, out_dir: pathlib.Path) -> None:
            self.mode = mode
            self.out_dir = out_dir

    class FakeResult:
        """Minimal stand-in for the workflow result dataclass."""

        ok = True

    fake_module.VerificationConfig = FakeConfig
    fake_module.run_verification = lambda _config: FakeResult()
    monkeypatch.setitem(sys.modules, "archimono.verification", fake_module)
    for runner_name in ("run_geometry", "run_exact_solver", "run_visualize"):
        monkeypatch.setattr(
            verify,
            runner_name,
            lambda _out_dir: pytest.fail(
                "CLI run must delegate to workflow module"
            ),
            raising=False,
        )

    args = argparse.Namespace(mode="all", out_dir=tmp_path)

    assert verify.run(args) == 0
