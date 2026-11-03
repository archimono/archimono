"""Sample-run report rendering: read the runs DB and write markdown."""

from __future__ import annotations

import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from archimono.assignment import SOLVERS_BY_NAME
from archimono.benchmarking import (
    SWEEP_CONFIGS,
    SolveRun,
    SolveRunDB,
    SweepConfig,
)
from archimono.certification import (
    FRUSTRATED_TILING_KEYS as _FRUSTRATED_TILING_KEYS,
)
from archimono.workflows._benchmark_common import _ALL_TILINGS, _print

# CLI short-name -> solver class-name string (as stored in the runs DB),
# derived from the canonical assignment.SOLVERS_BY_NAME registry.
_SOLVER_CLI_TO_CLASS: dict[str, str] = {
    name: cls.__name__ for name, cls in SOLVERS_BY_NAME.items()
}

@dataclass(slots=True)
class SampleRunReportConfig:
    """Configuration for rendering sample-run markdown reports.

    Attributes:
        db_path: SQLite database path to read.
        out_path: Markdown output path.
        tiling: Optional tiling-key filter.
        solver: Optional solver-name filter.
        n: Optional vertex-count filter.
    """

    db_path: pathlib.Path
    out_path: pathlib.Path
    tiling: str | None
    solver: str | None
    n: int | None



@dataclass(slots=True)
class SampleRunReportResult:
    """Result from rendering sample-run reports.

    Attributes:
        report_path: Full report path written, or ``None`` if no report was
            written.
        greedy_hard_report_path: Greedy-hard report path written for sweep
            data, or ``None``.
        runs_rendered: Number of runs included in the full report.
        greedy_hard_key_count: Number of greedy-hard supercells found for sweep
            data, or ``None`` when no sweep labels were present.
    """

    report_path: pathlib.Path | None
    greedy_hard_report_path: pathlib.Path | None
    runs_rendered: int
    greedy_hard_key_count: int | None


def _sr_filter_runs(
    runs: list[SolveRun],
    *,
    tiling: str | None,
    solver: str | None,
    n: int | None,
) -> list[SolveRun]:
    """Return runs matching all non-None filter criteria.

    Args:
        runs: Full run list to filter.
        tiling: If set, keep only runs with this ``tiling_key`` (or ``"frustrated"``
            to keep all frustrated tilings).
        solver: If set, keep only runs with this ``solver``.
        n: If set, keep only runs with this ``n_vertices``.
    """
    result = runs
    if tiling is not None:
        if tiling.lower() == "frustrated":
            result = [r for r in result if r.tiling_key in _FRUSTRATED_TILING_KEYS]
        else:
            result = [r for r in result if r.tiling_key == tiling]
    if solver is not None:
        normalized_solver = _SOLVER_CLI_TO_CLASS.get(solver.lower(), solver)
        result = [r for r in result if r.solver == normalized_solver]
    if n is not None:
        result = [r for r in result if r.n_vertices == n]
    return result


def _sr_detect_sweep_labels(runs: list[SolveRun]) -> list[str]:
    """Return sweep-config labels present in runs, in canonical SWEEP_CONFIGS order."""
    present = {r.sweep_label for r in runs if r.sweep_label is not None}
    return [cfg.label for cfg in SWEEP_CONFIGS if cfg.label in present]


def _sr_sweep_config_map() -> dict[str, SweepConfig]:
    """Return a label-keyed dict of all SWEEP_CONFIGS."""
    return {cfg.label: cfg for cfg in SWEEP_CONFIGS}


def _sr_table_group_sort_key(
    item: tuple[str, int, str],
) -> tuple[int, str, str]:
    """Sort table rows by nodes, then supercell config, then solver.

    Args:
        item: ``(solver, n_vertices, supercell)`` triple.

    Returns:
        ``(n_vertices, supercell, solver)`` sort key.
    """
    solver, n_vertices, supercell = item
    return (n_vertices, supercell, solver)


def _sr_render_sweep_header(sweep_labels: list[str]) -> list[str]:
    """Build the sweep-config description block for the report header."""
    cfg_map = _sr_sweep_config_map()
    lines: list[str] = ["## Sweep configurations", ""]
    lines.append(
        "| config | greedy_n_runs | annealing_n_runs"
        " | annealing_steps_per_temperature |"
    )
    lines.append("|---|---:|---:|---:|")
    for label in sweep_labels:
        cfg = cfg_map.get(label)
        if cfg is not None:
            lines.append(
                f"| **{label}** | {cfg.greedy_n_runs}"
                f" | {cfg.annealing_n_runs}"
                f" | {cfg.annealing_steps_per_temperature} |"
            )
        else:
            lines.append(f"| **{label}** | — | — | — |")
    lines.append("")
    lines.append("**Columns**")
    lines.append("")
    lines.append("- `solver`: algorithm used (GreedySolver or AnnealingSolver)")
    lines.append("- `n`: target supercell size (number of vertices)")
    lines.append("- `supercell`: supercell matrix in `(axb)` or `[a b;c d]` notation")
    lines.append("- `# runs`: independent single-restart samples collected")
    lines.append(
        "- `best_eff`: highest cut efficiency (cut edges / total edges)"
        " observed across all runs"
    )
    lines.append("- `not_max`: number of runs that did not reach `best_eff`")
    lines.append("- `avg_ms`: mean wall-clock runtime per run in milliseconds")
    return lines


def _sr_render_sweep_tiling_section(
    tiling_runs: list[SolveRun],
    *,
    sweep_labels: list[str],
    heading: str,
) -> list[str]:
    """Build a compact sweep-comparison table for one tiling.

    One row per ``(solver, n_vertices, supercell)``. Columns: the fixed key
    fields plus one data cell per detected sweep config.
    """
    labels = sweep_labels
    label_runs: dict[str, list[SolveRun]] = {
        label: [r for r in tiling_runs if r.sweep_label == label]
        for label in labels
    }
    keys: set[tuple[str, int, str]] = set()
    for runs_for_label in label_runs.values():
        for r in runs_for_label:
            keys.add((r.solver, r.n_vertices, r.supercell or ""))

    lines: list[str] = [heading, ""]
    n_data_cols = 4
    umbrella_cells = "".join(
        f'<th colspan="{n_data_cols}">{lbl}</th>' for lbl in labels
    )
    sub_cells = "".join(
        "<th>#&nbsp;runs</th><th>best_eff</th><th>not_max</th><th>avg_ms</th>"
        for _ in labels
    )
    lines.append(
        "<table><thead><tr>"
        "<th></th><th></th><th></th>"
        f"{umbrella_cells}"
        "</tr><tr>"
        "<th>solver</th><th>n</th><th>supercell</th>"
        f"{sub_cells}"
        "</tr></thead><tbody>"
    )
    for solver, n_vertices, supercell_str in sorted(
        keys, key=lambda x: (x[1], x[2], x[0])
    ):
        supercell_label = supercell_str or "-"
        row_cells: list[str] = [
            f"<td>{solver}</td><td>{n_vertices}</td><td>{supercell_label}</td>"
        ]
        for label in labels:
            group = [
                r
                for r in label_runs[label]
                if r.solver == solver
                and r.n_vertices == n_vertices
                and (r.supercell or "") == supercell_str
            ]
            ok = [
                r for r in group if r.status == "ok" and r.cut_efficiency is not None
            ]
            if not ok:
                row_cells.append("<td>-</td><td>-</td><td>-</td><td>-</td>")
                continue
            effs: list[float] = [
                r.cut_efficiency for r in ok if r.cut_efficiency is not None
            ]
            best_eff = float(np.max(effs))
            not_at_max = sum(1 for e in effs if e < best_eff)
            avg_rt = float(np.mean([r.runtime_ms for r in ok]))
            row_cells.append(
                f"<td>{len(ok)}</td><td>{best_eff:.4f}</td>"
                f"<td>{not_at_max}</td><td>{avg_rt:.0f}</td>"
            )
        lines.append(f"<tr>{''.join(row_cells)}</tr>")
    lines.append("</tbody></table>")
    return lines


def _sr_render_plain_tiling_section(
    tiling_runs: list[SolveRun],
    *,
    heading: str,
) -> list[str]:
    """Build a plain aggregate table for one tiling (no sweep)."""
    lines: list[str] = [heading, ""]
    lines.append(
        "| solver | n | supercell | #runs | best_eff | not_at_max | avg_rt_ms |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    groups: dict[tuple[str, int, str], list[SolveRun]] = defaultdict(list)
    for r in tiling_runs:
        groups[(r.solver, r.n_vertices, r.supercell or "")].append(r)
    for (solver, n_vertices, supercell_str), group in sorted(
        groups.items(), key=lambda item: _sr_table_group_sort_key(item[0])
    ):
        ok = [
            r for r in group if r.status == "ok" and r.cut_efficiency is not None
        ]
        supercell_label = supercell_str or "-"
        if not ok:
            lines.append(
                f"| {solver} | {n_vertices} | {supercell_label}"
                f" | {len(group)} | - | - | - |"
            )
            continue
        effs = [r.cut_efficiency for r in ok if r.cut_efficiency is not None]
        best_eff = float(np.max(effs))
        not_at_max = sum(1 for e in effs if e < best_eff)
        avg_rt = float(np.mean([r.runtime_ms for r in ok]))
        lines.append(
            f"| {solver} | {n_vertices} | {supercell_label}"
            f" | {len(ok)} | {best_eff:.4f} | {not_at_max} | {avg_rt:.0f} |"
        )
    return lines


def _sr_render_tiling_sections(
    runs: list[SolveRun],
    *,
    sweep_labels: list[str] | None,
    stream: TextIO | None,
) -> list[str]:
    """Build markdown lines for all tiling sections.

    Args:
        runs: Run records spanning all tilings.
        sweep_labels: If set, use the compact sweep-comparison table layout.
        stream: Optional progress-output stream.
    """
    tiling_order = [key for key, _ in _ALL_TILINGS]
    present_keys = {r.tiling_key for r in runs}
    ordered_keys = [k for k in tiling_order if k in present_keys]
    ordered_keys += sorted(present_keys - set(tiling_order))
    display_map = dict(_ALL_TILINGS)
    lines: list[str] = []
    for i, tiling_key in enumerate(ordered_keys, 1):
        tiling_runs = [r for r in runs if r.tiling_key == tiling_key]
        display = display_map.get(tiling_key, tiling_key)
        heading = f"## {display}"
        _print(
            stream,
            f"  [{i}/{len(ordered_keys)}] {display} ({len(tiling_runs)} runs) …",
            flush=True,
        )
        if sweep_labels:
            section = _sr_render_sweep_tiling_section(
                tiling_runs, sweep_labels=sweep_labels, heading=heading,
            )
        else:
            section = _sr_render_plain_tiling_section(tiling_runs, heading=heading)
        lines.extend(section)
        lines.append("")
    return lines


def _sr_greedy_hard_keys(
    runs: list[SolveRun],
    sweep_labels: list[str],
    *,
    ref_label: str = "default",
) -> set[tuple[str, int, str]]:
    """Return (tiling_key, n_vertices, supercell) triples where GreedySolver struggles.

    Args:
        runs: All run records.
        sweep_labels: Ordered labels of detected sweep configs.
        ref_label: Sweep-config label used as the reference for the filter.
    """
    if ref_label in sweep_labels:
        label: str | None = ref_label
    elif sweep_labels:
        label = sweep_labels[0]
    else:
        label = None
    if label is None:
        return set()

    ref_runs = [r for r in runs if r.sweep_label == label]
    groups: dict[tuple[str, int, str, str], list[SolveRun]] = defaultdict(list)
    for r in ref_runs:
        groups[(r.tiling_key, r.n_vertices, r.supercell or "", r.solver)].append(r)

    hard: set[tuple[str, int, str]] = set()
    supercell_keys = {(tk, nv, sc) for tk, nv, sc, _ in groups}
    for tiling_key, n_vertices, supercell in supercell_keys:
        greedy_runs = [
            r
            for r in groups[(tiling_key, n_vertices, supercell, "GreedySolver")]
            if r.status == "ok" and r.cut_efficiency is not None
        ]
        if not greedy_runs:
            continue
        g_effs: list[float] = [
            r.cut_efficiency for r in greedy_runs if r.cut_efficiency is not None
        ]
        g_best = float(np.max(g_effs))
        g_not_max = sum(1 for e in g_effs if e < g_best)
        high_failure = (g_not_max / len(greedy_runs)) > 0.5
        annealing_runs = [
            r
            for r in groups[(tiling_key, n_vertices, supercell, "AnnealingSolver")]
            if r.status == "ok" and r.cut_efficiency is not None
        ]
        a_effs: list[float] = [
            r.cut_efficiency for r in annealing_runs if r.cut_efficiency is not None
        ]
        worse_than_annealing = bool(a_effs and float(np.max(a_effs)) > g_best)
        if high_failure or worse_than_annealing:
            hard.add((tiling_key, n_vertices, supercell))
    return hard


def _sr_render_report_markdown(
    runs: list[SolveRun],
    *,
    db_path: pathlib.Path,
    sweep_labels: list[str] | None = None,
    stream: TextIO | None,
) -> str:
    """Build a markdown report string (no trailing newline) from *runs*.

    Args:
        runs: Filtered run records to render.
        db_path: Database path, embedded in the report header.
        sweep_labels: If set, ordered sweep-config labels detected in the runs.
        stream: Optional progress-output stream.
    """
    lines: list[str] = [
        "# Solver Run Report",
        "",
        f"**Source:** `{db_path}`  **Runs:** {len(runs)}",
        "",
    ]
    if sweep_labels:
        lines.extend(_sr_render_sweep_header(sweep_labels))
        lines.append("")
        lines.append("## Results")
        lines.append("")
    lines.extend(
        _sr_render_tiling_sections(
            runs,
            sweep_labels=sweep_labels,
            stream=stream,
        )
    )
    return "\n".join(lines)


def write_sample_run_report(
    config: SampleRunReportConfig,
    *,
    stream: TextIO | None = None,
) -> SampleRunReportResult:
    """Load runs from the database and write markdown reports.

    Writes a full report to ``config.out_path`` and a greedy-hard filtered report
    alongside it. Auto-detects sweep configs and groups by config when found.
    """
    _print(stream, f"Loading runs from {config.db_path} …", flush=True)
    db = SolveRunDB(config.db_path)
    all_runs = db.all()
    db.close()
    _print(stream, f"Loaded {len(all_runs)} runs.", flush=True)
    if not all_runs:
        _print(stream, "No runs found.")
        return SampleRunReportResult(
            report_path=None,
            greedy_hard_report_path=None,
            runs_rendered=0,
            greedy_hard_key_count=None,
        )

    runs = _sr_filter_runs(
        all_runs,
        tiling=config.tiling,
        solver=config.solver,
        n=config.n,
    )
    if not runs:
        _print(stream, "No runs matched the specified filters.")
        return SampleRunReportResult(
            report_path=None,
            greedy_hard_report_path=None,
            runs_rendered=0,
            greedy_hard_key_count=None,
        )
    _print(stream, f"Filtered to {len(runs)} runs.", flush=True)

    sweep_labels = _sr_detect_sweep_labels(runs) or None
    if sweep_labels:
        _print(stream, f"Detected sweep configs: {sweep_labels}", flush=True)

    _print(stream, "Rendering full report …", flush=True)
    md = _sr_render_report_markdown(
        runs,
        db_path=config.db_path,
        sweep_labels=sweep_labels,
        stream=stream,
    )
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text(md + "\n", encoding="utf-8")
    _print(stream, f"Report written to {config.out_path}", flush=True)

    greedy_hard_report_path: pathlib.Path | None = None
    greedy_hard_key_count: int | None = None
    if sweep_labels:
        _print(stream, "Computing greedy-hard supercells …", flush=True)
        hard_keys = _sr_greedy_hard_keys(runs, sweep_labels)
        greedy_hard_key_count = len(hard_keys)
        hard_runs = [
            r
            for r in runs
            if (r.tiling_key, r.n_vertices, r.supercell or "") in hard_keys
        ]
        _print(
            stream,
            f"Found {len(hard_keys)} greedy-hard supercells"
            f" ({len(hard_runs)} runs). Rendering …",
            flush=True,
        )
        hard_md = _sr_render_report_markdown(
            hard_runs,
            db_path=config.db_path,
            sweep_labels=sweep_labels,
            stream=stream,
        )
        greedy_hard_report_path = config.out_path.with_stem(
            config.out_path.stem + "-greedy-hard"
        )
        greedy_hard_report_path.write_text(hard_md + "\n", encoding="utf-8")
        _print(
            stream,
            f"Greedy-hard report written to {greedy_hard_report_path}",
            flush=True,
        )

    return SampleRunReportResult(
        report_path=config.out_path,
        greedy_hard_report_path=greedy_hard_report_path,
        runs_rendered=len(runs),
        greedy_hard_key_count=greedy_hard_key_count,
    )
