"""Fast parser-to-workflow adapter tests for ``archimono certify``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

import archimono
import archimono.workflows
from archimono.cli import __main__, certify

pytestmark = pytest.mark.fast


@dataclass
class _CertificationWorkflowConfig:
    """Complete stand-in for solver-certification workflow configuration."""

    solver: str
    json_out_dir: Path
    markdown_out: Path
    target_sizes: list[int]
    tilings: list[str]
    from_json_dir: Path | None
    recursive: bool
    one_supercell: bool
    memory_limit_gb: float
    bruteforce_md: Path


def _install_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_CertificationWorkflowConfig]:
    """Install capture-only certification dependencies at both lazy seams."""
    captured: list[_CertificationWorkflowConfig] = []
    certification = ModuleType("archimono.certification")

    def resolve_target_sizes(
        requested: list[int] | None, *, default: list[int], max_n: int
    ) -> list[int]:
        """Mirror target filtering without importing the real certification module."""
        if requested is not None:
            return requested
        return [size for size in default if size <= max_n]

    certification.resolve_target_sizes = resolve_target_sizes
    workflows = ModuleType("archimono.workflows.certify")
    workflows.CertificationWorkflowConfig = _CertificationWorkflowConfig
    workflows.run_certification_workflow = (
        lambda config, stream: captured.append(config)
    )
    monkeypatch.setitem(sys.modules, "archimono.certification", certification)
    monkeypatch.setitem(sys.modules, "archimono.workflows.certify", workflows)
    monkeypatch.setattr(archimono, "certification", certification, raising=False)
    monkeypatch.setattr(
        archimono.workflows,
        "certify",
        workflows,
        raising=False,
    )
    return captured


def test_certify_uses_the_bruteforce_default_targets_and_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bruteforce defaults retain their checkout-relative artifact locations."""
    captured = _install_workflow(monkeypatch)

    assert __main__.main(["certify", "--solver", "bruteforce"]) == 0
    config = captured[0]
    assert config.target_sizes == [12, 24]
    assert config.tilings == ["all"]
    assert config.json_out_dir == (
        certify._PROJECT_ROOT / "tmp" / "bruteforce-certification"
    )
    assert config.markdown_out == (
        certify._PROJECT_ROOT / "tmp" / "bruteforce-certification.md"
    )
    assert config.bruteforce_md == (
        certify._PROJECT_ROOT / "docs" / "draft" / "bruteforce-certification.md"
    )


def test_certify_uses_the_frontier_dp_value_default_targets_and_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Value-only frontier-DP retains its larger target sequence and paths."""
    captured = _install_workflow(monkeypatch)

    assert __main__.main(["certify", "--solver", "frontier-dp-value"]) == 0
    config = captured[0]
    assert config.target_sizes == [12, 24, 36, 48, 60]
    assert config.tilings == ["all"]
    assert config.json_out_dir == certify._PROJECT_ROOT / "tmp" / "certify-max-cut"
    assert config.markdown_out == (
        certify._PROJECT_ROOT / "tmp" / "certified-max-cut.md"
    )


def test_certify_uses_the_annealing_default_targets_and_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Annealing retains its default corpus, target sequence, and paths."""
    captured = _install_workflow(monkeypatch)

    assert __main__.main(["certify", "--solver", "annealing"]) == 0
    config = captured[0]
    assert config.target_sizes == [12, 24, 36, 48, 60]
    assert config.tilings == ["all"]
    assert config.json_out_dir == certify._PROJECT_ROOT / "tmp" / "annealing-stress"
    assert config.markdown_out == (
        certify._PROJECT_ROOT / "tmp" / "stress-test-results.md"
    )


def test_certify_bare_from_json_dir_uses_the_solver_json_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare replay flag selects the selected solver's default JSON directory."""
    captured = _install_workflow(monkeypatch)

    assert __main__.main(
        ["certify", "--solver", "frontier-dp-value", "--from-json-dir"]
    ) == 0
    config = captured[0]
    assert config.from_json_dir == config.json_out_dir


def test_certify_explicit_from_json_dir_and_recursive_flag_reach_the_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflow(monkeypatch)

    assert __main__.main(
        [
            "certify",
            "--solver",
            "bruteforce",
            "--from-json-dir",
            "replay-json",
            "--recursive",
        ]
    ) == 0
    config = captured[0]
    assert config.from_json_dir == Path("replay-json")
    assert config.recursive is True
