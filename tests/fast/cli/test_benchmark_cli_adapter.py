"""Fast parser-to-workflow adapter tests for ``archimono benchmark``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

import archimono
import archimono.workflows
from archimono.cli import __main__, benchmark

pytestmark = pytest.mark.fast


@dataclass
class _FrontierOrderConfig:
    """Complete stand-in for frontier-order benchmark configuration."""

    target_sizes: list[int] | None
    max_n: int
    tilings: list[str] | None
    md_out: Path


@dataclass
class _NumbaJitConfig:
    """Stand-in for the empty Numba-JIT benchmark configuration."""


@dataclass
class _SampleRunsConfig:
    """Complete stand-in for sampling benchmark configuration."""

    db_path: Path
    target_sizes: list[int] | None
    max_n: int
    tilings: list[str] | None
    solvers: list[str]
    n_runs: int
    seed: int
    annealing_temperature: float
    annealing_min_temperature: float
    annealing_cooling_rate: float
    annealing_steps_per_temperature: int
    sweep: bool
    replace_existing: bool


@dataclass
class _SampleRunReportConfig:
    """Complete stand-in for sample-run report configuration."""

    db_path: Path
    out_path: Path
    tiling: str | None
    solver: str | None
    n: int | None


def _install_workflows(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[object]]:
    """Install all capture-only benchmark workflow functions at their lazy seam."""
    captured: dict[str, list[object]] = {
        "frontier_order": [],
        "numba_jit": [],
        "sample_runs": [],
        "report": [],
    }
    workflows = ModuleType("archimono.workflows.benchmark")
    workflows.FrontierOrderConfig = _FrontierOrderConfig
    workflows.NumbaJitConfig = _NumbaJitConfig
    workflows.SampleRunsConfig = _SampleRunsConfig
    workflows.SampleRunReportConfig = _SampleRunReportConfig
    workflows.run_frontier_order = lambda config: captured["frontier_order"].append(
        config
    )
    workflows.run_numba_jit = lambda config: captured["numba_jit"].append(config)
    workflows.run_sample_runs = lambda config: captured["sample_runs"].append(config)
    workflows.write_sample_run_report = lambda config: captured["report"].append(config)
    monkeypatch.setitem(sys.modules, "archimono.workflows.benchmark", workflows)
    monkeypatch.setattr(
        archimono.workflows, "benchmark", workflows, raising=False
    )
    return captured


def test_frontier_order_benchmark_options_create_a_frontier_order_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflows(monkeypatch)

    assert __main__.main(
        [
            "benchmark",
            "frontier-order",
            "--target-sizes",
            "36",
            "--max-n",
            "48",
            "--tilings",
            "square",
            "--md-out",
            "order.md",
        ]
    ) == 0
    assert captured["frontier_order"] == [
        _FrontierOrderConfig([36], 48, ["square"], Path("order.md"))
    ]


def test_numba_jit_import_error_is_reported_to_stderr_with_status_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = _install_workflows(monkeypatch)
    workflows = sys.modules["archimono.workflows.benchmark"]

    def fail(config: _NumbaJitConfig) -> None:
        captured["numba_jit"].append(config)
        raise ImportError("numba is unavailable")

    workflows.run_numba_jit = fail

    assert __main__.main(["benchmark", "numba-jit"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "numba is unavailable" in output.err


def test_sample_runs_options_create_a_sampling_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflows(monkeypatch)

    assert __main__.main(
        [
            "benchmark",
            "sample-runs",
            "--db",
            "runs.sqlite",
            "--target-sizes",
            "12",
            "--max-n",
            "24",
            "--tilings",
            "kagome",
            "--solvers",
            "greedy",
            "--n-runs",
            "3",
            "--seed",
            "7",
            "--sweep",
            "--replace-existing",
        ]
    ) == 0
    config = captured["sample_runs"][0]
    assert config.db_path == Path("runs.sqlite")
    assert config.target_sizes == [12]
    assert config.max_n == 24
    assert config.tilings == ["kagome"]
    assert config.solvers == ["greedy"]
    assert config.n_runs == 3
    assert config.seed == 7
    assert config.sweep is True
    assert config.replace_existing is True


def test_sample_runs_report_creates_a_report_config_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report mode dispatches only to report generation and retains default paths."""
    captured = _install_workflows(monkeypatch)
    workflows = sys.modules["archimono.workflows.benchmark"]
    workflows.run_sample_runs = lambda config: pytest.fail("sampling must not run")

    assert __main__.main(
        ["benchmark", "sample-runs", "--db", "runs.sqlite", "--report"]
    ) == 0
    config = captured["report"][0]
    assert config.db_path == Path("runs.sqlite")
    assert config.out_path == (
        benchmark._PROJECT_ROOT / "docs" / "draft" / "solver-characterization.md"
    )
    assert captured["sample_runs"] == []


def test_benchmark_default_paths_are_checkout_relative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_workflows(monkeypatch)

    assert __main__.main(["benchmark", "frontier-order"]) == 0
    assert captured["frontier_order"][0].md_out == (
        benchmark._PROJECT_ROOT
        / "docs"
        / "draft"
        / "frontier-order-comparison.md"
    )
