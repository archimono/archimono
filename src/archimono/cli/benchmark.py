"""CLI parser and adapter for ``archimono benchmark``."""

from __future__ import annotations

import argparse as _argparse
import pathlib as _pathlib
import sys as _sys
from collections.abc import Callable as _Callable
from typing import cast as _cast

from archimono import _paths

__all__ = ["configure_parser", "run"]

_PROJECT_ROOT = _paths.PROJECT_ROOT
_SR_DEFAULT_SIZES: list[int] = list(range(12, 101, 12))


def _run_frontier_order(args: _argparse.Namespace) -> int | None:
    """Run the frontier-order benchmark from parsed arguments."""
    from archimono.workflows import benchmark as benchmark_workflows

    benchmark_workflows.run_frontier_order(
        benchmark_workflows.FrontierOrderConfig(
            target_sizes=args.target_sizes,
            max_n=args.max_n,
            tilings=args.tilings,
            md_out=args.md_out,
        )
    )
    return None


def _run_numba_jit(args: _argparse.Namespace) -> int | None:
    """Run the numba-JIT benchmark; return 1 on ImportError, else None."""
    from archimono.workflows import benchmark as benchmark_workflows

    try:
        benchmark_workflows.run_numba_jit(benchmark_workflows.NumbaJitConfig())
    except ImportError as exc:
        print(str(exc), file=_sys.stderr)
        return 1
    return None


def _run_sample_runs(args: _argparse.Namespace) -> int | None:
    """Write the sample-run report if ``--report`` is set; otherwise sample runs."""
    from archimono.workflows import benchmark as benchmark_workflows

    if args.report:
        benchmark_workflows.write_sample_run_report(
            benchmark_workflows.SampleRunReportConfig(
                db_path=args.db,
                out_path=args.report_out,
                tiling=args.report_tiling,
                solver=args.report_solver,
                n=args.report_n,
            )
        )
        return None

    benchmark_workflows.run_sample_runs(
        benchmark_workflows.SampleRunsConfig(
            db_path=args.db,
            target_sizes=args.target_sizes,
            max_n=args.max_n,
            tilings=args.tilings,
            solvers=args.solvers,
            n_runs=args.n_runs,
            seed=args.seed,
            annealing_temperature=args.annealing_temperature,
            annealing_min_temperature=args.annealing_min_temperature,
            annealing_cooling_rate=args.annealing_cooling_rate,
            annealing_steps_per_temperature=args.annealing_steps_per_temperature,
            sweep=args.sweep,
            replace_existing=args.replace_existing,
        )
    )
    return None


def configure_parser(subparser: _argparse.ArgumentParser) -> None:
    """Configure the ``archimono benchmark`` subparser."""
    subparser.description = (
        "Benchmarking tools for archimono solvers and frontier DP."
    )
    sub = subparser.add_subparsers(dest="subcommand", required=True)

    p_fo = sub.add_parser(
        "frontier-order",
        help="Compare vertex-ordering heuristics by frontier width.",
    )
    p_fo.add_argument(
        "--target-sizes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Target sizes to benchmark. Defaults to "
            "[12, 24, 36, 48, 60, 72, 84, 96]."
        ),
    )
    p_fo.add_argument(
        "--max-n",
        type=int,
        default=100,
        help="Upper bound on target sizes when --target-sizes is omitted.",
    )
    p_fo.add_argument(
        "--tilings",
        nargs="*",
        default=None,
        help="Subset of tiling keys. Defaults to all 11 Archimedean tilings.",
    )
    p_fo.add_argument(
        "--md-out",
        type=_pathlib.Path,
        default=_PROJECT_ROOT / "docs" / "draft" / "frontier-order-comparison.md",
        help="Path to write the markdown report.",
    )
    p_fo.set_defaults(_benchmark_run=_run_frontier_order)

    p_jit = sub.add_parser(
        "numba-jit",
        help="Benchmark @njit hot-path kernels vs pure Python.",
    )
    p_jit.set_defaults(_benchmark_run=_run_numba_jit)

    p_sr = sub.add_parser(
        "sample-runs",
        help="Sample per-restart solver runs and write to a SQLite database.",
    )
    p_sr.add_argument(
        "--db",
        type=_pathlib.Path,
        default=None,
        metavar="PATH",
        required=True,
        help="SQLite database path (required).",
    )
    p_sr.add_argument(
        "--target-sizes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit target sizes. Defaults to the 12-step corpus "
            f"{_SR_DEFAULT_SIZES} truncated to --max-n."
        ),
    )
    p_sr.add_argument(
        "--max-n",
        type=int,
        default=100,
        help="Upper bound on target sizes when --target-sizes is omitted.",
    )
    p_sr.add_argument(
        "--tilings",
        nargs="*",
        default=None,
        help="Subset of tiling keys. Defaults to all 11 Archimedean tilings.",
    )
    p_sr.add_argument(
        "--solvers",
        nargs="*",
        choices=["greedy", "annealing"],
        default=["greedy", "annealing"],
        help="Which solvers to sample. Defaults to both.",
    )
    p_sr.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Independent single-restart samples per (tiling, n, solver) point.",
    )
    p_sr.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; run i uses seed + i.",
    )
    p_sr.add_argument(
        "--annealing-temperature",
        type=float,
        default=5.0,
        help="AnnealingSolver initial temperature.",
    )
    p_sr.add_argument(
        "--annealing-min-temperature",
        type=float,
        default=1e-4,
        help="AnnealingSolver min_temperature.",
    )
    p_sr.add_argument(
        "--annealing-cooling-rate",
        type=float,
        default=0.995,
        help="AnnealingSolver cooling_rate.",
    )
    p_sr.add_argument(
        "--annealing-steps-per-temperature",
        type=int,
        default=200,
        help="AnnealingSolver steps_per_temperature.",
    )
    p_sr.add_argument(
        "--sweep",
        action="store_true",
        help="Run the built-in fast/default/thorough sampling presets.",
    )
    p_sr.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace existing DB rows instead of skipping them.",
    )
    p_sr.add_argument(
        "--report",
        action="store_true",
        default=False,
        help=(
            "Load runs from --db, write markdown reports, and exit. "
            "No solving is performed."
        ),
    )
    p_sr.add_argument(
        "--report-out",
        type=_pathlib.Path,
        default=_PROJECT_ROOT / "docs" / "draft" / "solver-characterization.md",
        metavar="PATH",
        help="Destination for the markdown file written by --report.",
    )
    p_sr.add_argument(
        "--report-tiling",
        default=None,
        metavar="KEY",
        help="Used with --report: restrict output to this tiling key.",
    )
    p_sr.add_argument(
        "--report-solver",
        default=None,
        metavar="NAME",
        help="Used with --report: restrict output to this solver name.",
    )
    p_sr.add_argument(
        "--report-n",
        type=int,
        default=None,
        metavar="N",
        help="Used with --report: restrict output to this n_vertices.",
    )
    p_sr.set_defaults(_benchmark_run=_run_sample_runs)


def run(args: _argparse.Namespace) -> int | None:
    """Dispatch to the selected ``benchmark`` subcommand."""
    benchmark_run = _cast(
        _Callable[[_argparse.Namespace], int | None],
        args._benchmark_run,
    )
    return benchmark_run(args)
