"""Fast tests for non-CLI benchmark workflow APIs."""

from __future__ import annotations

import dataclasses
import io
import pathlib

import networkx as nx

from archimono.benchmarking import SolveRunDB


def test_sample_runs_workflow_writes_runs_without_argparse_namespace(
    tmp_path: pathlib.Path,
) -> None:
    from archimono.workflows.benchmark import SampleRunsConfig, run_sample_runs

    db_path = tmp_path / "runs.db"
    stream = io.StringIO()

    result = run_sample_runs(
        SampleRunsConfig(
            db_path=db_path,
            target_sizes=[12],
            max_n=12,
            tilings=["hexagonal"],
            solvers=["greedy"],
            n_runs=1,
            seed=0,
            annealing_temperature=5.0,
            annealing_min_temperature=1e-4,
            annealing_cooling_rate=0.995,
            annealing_steps_per_temperature=200,
            sweep=False,
            replace_existing=False,
        ),
        stream=stream,
    )

    db = SolveRunDB(db_path)
    try:
        runs = db.all()
    finally:
        db.close()

    assert result.runs_written == len(runs)
    assert len(runs) > 0
    assert {run.tiling_key for run in runs} == {"hexagonal"}
    assert {run.solver for run in runs} == {"GreedySolver"}
    assert "DB opened at" in stream.getvalue()


def test_report_workflow_renders_markdown_without_argparse_namespace(
    tmp_path: pathlib.Path,
) -> None:
    from archimono.workflows.benchmark import (
        SampleRunReportConfig,
        SampleRunsConfig,
        run_sample_runs,
        write_sample_run_report,
    )

    db_path = tmp_path / "runs.db"
    report_path = tmp_path / "report.md"

    run_sample_runs(
        SampleRunsConfig(
            db_path=db_path,
            target_sizes=[12],
            max_n=12,
            tilings=["hexagonal"],
            solvers=["greedy"],
            n_runs=1,
            seed=0,
            annealing_temperature=5.0,
            annealing_min_temperature=1e-4,
            annealing_cooling_rate=0.995,
            annealing_steps_per_temperature=200,
            sweep=False,
            replace_existing=False,
        ),
        stream=io.StringIO(),
    )

    result = write_sample_run_report(
        SampleRunReportConfig(
            db_path=db_path,
            out_path=report_path,
            tiling=None,
            solver=None,
            n=None,
        ),
        stream=io.StringIO(),
    )

    assert result.report_path == report_path
    assert result.greedy_hard_report_path is None
    assert "# Solver Run Report" in report_path.read_text(encoding="utf-8")
    assert "GreedySolver" in report_path.read_text(encoding="utf-8")


def test_sample_run_solver_failures_are_captured_as_error_rows() -> None:
    """Solver failures are captured as status='error' run rows, not propagated.

    A single solver exception must not abort the whole batch (which would roll
    back every uncommitted row); it is recorded as an ``"error"`` ``SolveRun``.
    """
    from archimono.workflows.benchmark_sample_runs import (
        SampleRunsConfig,
        _sr_run_once,
    )

    config = SampleRunsConfig(
        db_path=pathlib.Path(":memory:"),
        target_sizes=[12],
        max_n=12,
        tilings=["hexagonal"],
        solvers=["greedy"],
        n_runs=1,
        seed=0,
        annealing_temperature=5.0,
        annealing_min_temperature=1e-4,
        annealing_cooling_rate=0.995,
        annealing_steps_per_temperature=200,
        sweep=False,
        replace_existing=False,
    )

    run = _sr_run_once(
        config=config,
        solver_name="GreedySolver",
        run_index=0,
        seed=0,
        graph=nx.Graph(),
        tiling_key="empty",
        n_vertices=0,
        supercell="(0x0)",
        n_edges=0,
        sweep_label=None,
    )

    assert run.status == "error"
    assert "graph must contain at least one node" in run.note
    assert run.cut_value is None
    assert run.cut_efficiency is None


def test_frontier_order_markdown_renders_precomputed_memory() -> None:
    """The memory column uses the per-entry estimate computed at collection.

    Entries carry the reachability-aware estimate (in GB) computed while the
    graph was still in scope; the renderer must not re-derive it from k alone.
    """
    from archimono.workflows.benchmark_frontier_order import (
        _render_frontier_order_markdown,
    )

    widths = {"natural": 4, "ck": 3, "greedy": 3, "geometric": 3}
    sections = [("kagome", "kagome", 3, 4, [(12, "(2x2)", widths, 1.5)])]

    markdown = _render_frontier_order_markdown(sections)

    assert "| 1.5 GB |" in markdown


def test_workflow_configs_are_dataclasses() -> None:
    """Workflow entry points use typed config dataclasses, not CLI namespaces."""
    from archimono.workflows.benchmark import (
        FrontierOrderConfig,
        NumbaJitConfig,
        SampleRunReportConfig,
        SampleRunsConfig,
    )

    assert dataclasses.is_dataclass(FrontierOrderConfig)
    assert dataclasses.is_dataclass(NumbaJitConfig)
    assert dataclasses.is_dataclass(SampleRunReportConfig)
    assert dataclasses.is_dataclass(SampleRunsConfig)
