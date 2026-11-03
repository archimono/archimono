"""Sample-runs benchmark workflow: per-restart solver runs and reports."""

from __future__ import annotations

import pathlib
import sqlite3
import time
from dataclasses import dataclass, replace
from typing import Any, TextIO

import networkx as nx
import numpy as np
import numpy.typing as npt

from archimono.assignment import AnnealingSolver, GreedySolver
from archimono.benchmarking import (
    SWEEP_CONFIGS,
    SolveRun,
    insert_run,
    open_db,
)
from archimono.tilings import registry
from archimono.tilings.base import Tiling, valid_supercell_matrices
from archimono.workflows._benchmark_common import (
    _ALL_TILINGS,
    _print,
    _resolve_target_sizes,
    _selected_tilings,
)

_SPECIES: tuple[str, str] = ("A", "B")


@dataclass(slots=True)
class SampleRunsConfig:
    """Configuration for sampling per-restart solver runs.

    Attributes:
        db_path: SQLite database path.
        target_sizes: Explicit target sizes, or ``None`` to use the default
            sequence truncated by ``max_n``.
        max_n: Upper bound applied when ``target_sizes`` is ``None``.
        tilings: Optional tiling-key filter.  ``"frustrated"`` expands to the
            frustrated Archimedean tilings.
        solvers: Solver choices as CLI-level names: ``"greedy"`` and/or
            ``"annealing"``.
        n_runs: Independent single-restart samples per solver in non-sweep
            mode.
        seed: Base seed; run ``i`` uses ``seed + i``.
        annealing_temperature: Annealing initial temperature.
        annealing_min_temperature: Annealing minimum temperature.
        annealing_cooling_rate: Annealing cooling rate.
        annealing_steps_per_temperature: Annealing steps per temperature.
        sweep: Whether to run the built-in sweep presets.
        replace_existing: Whether duplicate run rows replace existing rows.
        greedy_n_runs: Internal sweep override for greedy run count.
        annealing_n_runs: Internal sweep override for annealing run count.
    """

    db_path: pathlib.Path
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
    greedy_n_runs: int | None = None
    annealing_n_runs: int | None = None


@dataclass(slots=True)
class SampleRunsResult:
    """Result from sampling per-restart solver runs.

    Attributes:
        db_path: SQLite database path written.
        runs_written: Number of run rows inserted or replaced.
    """

    db_path: pathlib.Path
    runs_written: int


def _format_supercell_diag(matrix: npt.NDArray[np.intp]) -> str:
    """Format a 2×2 supercell matrix as a short string.

    Unlike :func:`archimono.certification.format_supercell`, the shear branch
    emits ``[a b;c d]`` (no space, all four entries). These strings are
    persisted as SQLite ``supercell`` keys, so the divergent format is kept to
    avoid forking existing DB histories.

    Returns:
        String like ``(2x3)`` for diagonal matrices, or
        ``[a b;c d]`` for off-diagonal ones.
    """
    if int(matrix[0, 1]) == 0 and int(matrix[1, 0]) == 0:
        return f"({int(matrix[0, 0])}x{int(matrix[1, 1])})"
    return (
        f"[{int(matrix[0, 0])} {int(matrix[0, 1])};"
        f"{int(matrix[1, 0])} {int(matrix[1, 1])}]"
    )


_SR_DEFAULT_SIZES: list[int] = list(range(12, 101, 12))


def _n_runs_for_solver(config: SampleRunsConfig, solver_name: str) -> int:
    """Return the run count for *solver_name* given *config*.

    In sweep mode ``config`` carries ``greedy_n_runs`` / ``annealing_n_runs``
    set by :func:`_sr_run_sweep`; otherwise falls back to ``config.n_runs``.

    Args:
        config: Sampling workflow configuration.
        solver_name: ``"GreedySolver"`` or ``"AnnealingSolver"``.
    """
    if solver_name == "GreedySolver" and config.greedy_n_runs is not None:
        return config.greedy_n_runs
    if solver_name == "AnnealingSolver" and config.annealing_n_runs is not None:
        return config.annealing_n_runs
    return config.n_runs


def _make_greedy_solver(_config: SampleRunsConfig, seed: int) -> GreedySolver:
    """Construct a GreedySolver seeded with *seed*; *_config* is unused."""
    return GreedySolver(n_restarts=1, seed=seed)


def _make_annealing_solver(
    config: SampleRunsConfig,
    seed: int,
) -> AnnealingSolver:
    """Construct an AnnealingSolver with hyperparameters from *config*."""
    return AnnealingSolver(
        temperature=config.annealing_temperature,
        min_temperature=config.annealing_min_temperature,
        cooling_rate=config.annealing_cooling_rate,
        steps_per_temperature=config.annealing_steps_per_temperature,
        n_restarts=1,
        seed=seed,
    )


def _sr_run_once(
    *,
    config: SampleRunsConfig,
    solver_name: str,
    run_index: int,
    seed: int,
    graph: nx.Graph[int],
    tiling_key: str,
    n_vertices: int,
    supercell: str,
    n_edges: int,
    sweep_label: str | None,
) -> SolveRun:
    """Execute one single-restart solve and return a :class:`SolveRun`.

    Args:
        config: Sampling workflow configuration.
        solver_name: ``"GreedySolver"`` or ``"AnnealingSolver"``.
        run_index: Zero-based index of this run in the sampling batch.
        seed: RNG seed for this run.
        graph: Assignment graph to solve.
        tiling_key: Registry key of the tiling.
        n_vertices: Number of vertices in the graph.
        supercell: Formatted supercell string.
        n_edges: Number of edges in the graph.
        sweep_label: Sweep-config label if running under ``--sweep``,
            else ``None``.

    Returns:
        Populated :class:`SolveRun` with status ``"ok"`` on success or
        ``"error"`` on any exception (with ``note`` set to the exception
        message and result fields left null).
    """
    if solver_name == "GreedySolver":
        solver: GreedySolver | AnnealingSolver = _make_greedy_solver(config, seed)
    else:
        solver = _make_annealing_solver(config, seed)

    t0 = time.perf_counter()
    status = "ok"
    note = ""
    cut_value: float | None = None
    cut_efficiency: float | None = None
    n_frustrated: int | None = None
    labels: tuple[int, ...] | None = None
    metadata: dict[str, Any] = {}

    try:
        result = solver.solve(graph, list(_SPECIES))
        cut_value = (
            float(result.cut_value) if result.cut_value is not None else None
        )
        n_frustrated = result.n_frustrated
        labels = result.labels
        metadata = dict(result.metadata)
        if solver_name == "AnnealingSolver":
            metadata["temperature"] = config.annealing_temperature
            metadata["min_temperature"] = config.annealing_min_temperature
            metadata["cooling_rate"] = config.annealing_cooling_rate
        if cut_value is not None and n_edges > 0:
            cut_efficiency = cut_value / n_edges
    except Exception as exc:  # noqa: BLE001
        status, note = "error", str(exc)

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return SolveRun(
        tiling_key=tiling_key,
        solver=solver_name,
        run_index=run_index,
        seed=seed,
        n_vertices=n_vertices,
        supercell=supercell,
        n_edges=n_edges,
        cut_value=cut_value,
        cut_efficiency=cut_efficiency,
        n_frustrated=n_frustrated,
        labels=labels,
        runtime_ms=runtime_ms,
        status=status,
        note=note,
        metadata=metadata,
        sweep_label=sweep_label,
    )


def _sr_sample_point(
    *,
    config: SampleRunsConfig,
    tiling: Tiling,
    tiling_key: str,
    target_n: int,
    solver_names: list[str],
    sweep_label: str | None,
    db_conn: sqlite3.Connection,
    stream: TextIO | None,
) -> int:
    """Sample all runs for one (tiling, target_n) point; return the rows written."""
    matrices = valid_supercell_matrices(tiling, target_n)
    if not matrices:
        _print(
            stream,
            f"  n={target_n:>3}  skip: no valid supercell",
            flush=True,
        )
        return 0

    runs_written = 0
    solver_summary = ", ".join(solver_names)
    graph_id_cache: dict[tuple[str, str, int, int], int] = {}
    solver_config_id_cache: dict[tuple[Any, ...], int] = {}

    for matrix in matrices:
        supercell = _format_supercell_diag(matrix)
        graph = tiling.graph(matrix)
        n_vertices = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        for solver_name in solver_names:
            for run_index in range(_n_runs_for_solver(config, solver_name)):
                seed = config.seed + run_index
                run = _sr_run_once(
                    config=config,
                    solver_name=solver_name,
                    run_index=run_index,
                    seed=seed,
                    graph=graph,
                    tiling_key=tiling_key,
                    n_vertices=n_vertices,
                    supercell=supercell,
                    n_edges=n_edges,
                    sweep_label=sweep_label,
                )
                if insert_run(
                    db_conn,
                    run,
                    replace=config.replace_existing,
                    graph_id_cache=graph_id_cache,
                    solver_config_id_cache=solver_config_id_cache,
                ):
                    runs_written += 1

        total_runs_this_point = sum(
            _n_runs_for_solver(config, s) for s in solver_names
        )
        _print(
            stream,
            f"  n={n_vertices:>3}  supercell={supercell:>10}  "
            f"solvers=[{solver_summary}]  runs={total_runs_this_point}",
            flush=True,
        )
    return runs_written


def _sr_solver_names(config: SampleRunsConfig) -> list[str]:
    """Map CLI solver choices to solver class names."""
    names: list[str] = []
    if "greedy" in config.solvers:
        names.append("GreedySolver")
    if "annealing" in config.solvers:
        names.append("AnnealingSolver")
    return names


def _sr_run_one(
    config: SampleRunsConfig,
    *,
    sweep_label: str | None,
    db_conn: sqlite3.Connection,
    stream: TextIO | None,
) -> int:
    """Run one sampling configuration; return the DB rows inserted or replaced."""
    target_sizes = _resolve_target_sizes(
        config.target_sizes, default=_SR_DEFAULT_SIZES, max_n=config.max_n
    )
    selected = _selected_tilings(config.tilings, _ALL_TILINGS)
    solver_names = _sr_solver_names(config)

    total_runs = 0
    for tiling_key, display in selected:
        tiling = registry.get(tiling_key)
        _print(
            stream,
            f"\n[{display}]  n_v={tiling.n_vertices}  coord={tiling.coordination}  "
            f"bipartite={tiling.is_bipartite}",
            flush=True,
        )
        for target_n in target_sizes:
            total_runs += _sr_sample_point(
                config=config,
                tiling=tiling,
                tiling_key=tiling_key,
                target_n=target_n,
                solver_names=solver_names,
                sweep_label=sweep_label,
                db_conn=db_conn,
                stream=stream,
            )

    db_conn.commit()
    _print(stream, f"\nDone. {total_runs} runs committed to DB.", flush=True)
    return total_runs


def _sr_run_sweep(
    config: SampleRunsConfig,
    *,
    stream: TextIO | None,
) -> int:
    """Run all SWEEP_CONFIGS in sequence; return the DB rows inserted or replaced."""
    _print(
        stream,
        f"Starting sweep of {len(SWEEP_CONFIGS)} configurations",
        flush=True,
    )
    t_sweep_start = time.perf_counter()

    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    db_conn = open_db(config.db_path)
    _print(stream, f"DB opened at {config.db_path}", flush=True)

    total_runs = 0
    for cfg in SWEEP_CONFIGS:
        sweep_config = replace(
            config,
            greedy_n_runs=cfg.greedy_n_runs,
            annealing_n_runs=cfg.annealing_n_runs,
            annealing_steps_per_temperature=cfg.annealing_steps_per_temperature,
        )
        _print(
            stream,
            f"\n{'=' * 60}\n"
            f"Sweep config: {cfg.label!r}\n"
            f"  greedy_n_runs={cfg.greedy_n_runs}\n"
            f"  annealing_n_runs={cfg.annealing_n_runs}\n"
            f"  annealing_steps_per_temperature="
            f"{cfg.annealing_steps_per_temperature}\n"
            f"{'=' * 60}",
            flush=True,
        )
        t_cfg_start = time.perf_counter()
        total_runs += _sr_run_one(
            sweep_config,
            sweep_label=cfg.label,
            db_conn=db_conn,
            stream=stream,
        )
        elapsed = time.perf_counter() - t_cfg_start
        _print(
            stream,
            f"Config {cfg.label!r} done in {elapsed / 60:.1f} min.",
            flush=True,
        )

    db_conn.close()

    total_elapsed = time.perf_counter() - t_sweep_start
    _print(
        stream,
        f"\nSweep complete: {len(SWEEP_CONFIGS)} configs in "
        f"{total_elapsed / 60:.1f} min.",
        flush=True,
    )
    return total_runs


def run_sample_runs(
    config: SampleRunsConfig,
    *,
    stream: TextIO | None = None,
) -> SampleRunsResult:
    """Run the sample-runs workflow."""
    if config.sweep:
        runs_written = _sr_run_sweep(config, stream=stream)
    else:
        config.db_path.parent.mkdir(parents=True, exist_ok=True)
        db_conn = open_db(config.db_path)
        _print(stream, f"DB opened at {config.db_path}", flush=True)
        runs_written = _sr_run_one(
            config,
            sweep_label=None,
            db_conn=db_conn,
            stream=stream,
        )
        db_conn.close()
    return SampleRunsResult(db_path=config.db_path, runs_written=runs_written)
