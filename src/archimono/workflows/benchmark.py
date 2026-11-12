"""Reusable benchmark and report workflows for archimono.

Compatibility facade: each subcommand lives in its own module and is
re-exported here so existing imports keep working.

Workflows
---------
frontier-order
    Compare vertex-ordering heuristics by maximum frontier width
    (:mod:`archimono.workflows.benchmark_frontier_order`).

numba-jit
    Benchmark the @njit hot-path kernels against their pure-Python
    equivalents (:mod:`archimono.workflows.benchmark_numba_jit`).

sample-runs
    Sample per-restart solver runs across the Archimedean-tiling corpus and
    write results to a SQLite database
    (:mod:`archimono.workflows.benchmark_sample_runs`).
"""

from __future__ import annotations

from archimono.workflows.benchmark_frontier_order import (
    FrontierOrderConfig,
    FrontierOrderResult,
    run_frontier_order,
)
from archimono.workflows.benchmark_numba_jit import (
    NumbaJitConfig,
    NumbaJitResult,
    run_numba_jit,
)
from archimono.workflows.benchmark_sample_report import (
    SampleRunReportConfig,
    SampleRunReportResult,
    write_sample_run_report,
)
from archimono.workflows.benchmark_sample_runs import (
    SampleRunsConfig,
    SampleRunsResult,
    run_sample_runs,
)

__all__ = [
    "FrontierOrderConfig",
    "FrontierOrderResult",
    "NumbaJitConfig",
    "NumbaJitResult",
    "SampleRunReportConfig",
    "SampleRunReportResult",
    "SampleRunsConfig",
    "SampleRunsResult",
    "run_frontier_order",
    "run_numba_jit",
    "run_sample_runs",
    "write_sample_run_report",
]
