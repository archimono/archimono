"""Solver-certification workflows for all MAX-CUT solver backends.

Dispatches to one of three solver backends via ``--solver``:

- ``bruteforce``        — exhaustive fixed-composition enumeration
- ``frontier-dp-value`` — frontier DP value-only path (faster, no labels)
- ``annealing``         — simulated annealing (bipartite tilings use exact
                          2-colouring via BipartiteAssigner)

For each solver the workflow iterates over the selected tilings and target sizes,
writes one JSON file per solved case to ``--json-out-dir``, and writes a
markdown summary to ``--markdown-out``.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
import time
from typing import Any, Literal, TextIO, TypeAlias

import networkx as nx
from networkx.algorithms import bipartite as nx_bipartite

from archimono import _paths, assignment
from archimono.assignment import (
    AnnealingSolver,
    BipartiteAssigner,
    compute_frontier_sizes,
    compute_max_cut_value,
    estimate_peak_memory_bytes,
    select_heuristic,
)
from archimono.assignment.solvers.parallel_bruteforce import (
    solve_fixed_composition_parallel,
)
from archimono.certification import (
    TILINGS,
    CertificationRecord,
    build_bruteforce_record,
    format_supercell,
    load_case_json_dir,
    load_case_json_tree,
    make_skip_record,
    resolve_tilings,
    write_case_json,
    write_markdown_summary,
)
from archimono.tilings import base, registry

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SLOW_SOLVE_THRESHOLD_S: float = 300.0

_ANNEALING_SOLVER = AnnealingSolver(n_restarts=30, seed=42)
_PROJECT_ROOT = _paths.PROJECT_ROOT
_GENERATED_BY = "archimono certify"

SolverName: TypeAlias = Literal[
    "bruteforce",
    "frontier-dp-value",
    "annealing",
]


@dataclasses.dataclass(frozen=True, slots=True)
class CertificationWorkflowConfig:
    """Configuration for one solver-certification workflow run.

    Attributes:
        solver: Solver backend to use.
        json_out_dir: Directory for per-case JSON output.
        markdown_out: Path for markdown summary output.
        target_sizes: Target node counts to solve.
        tilings: Tiling registry keys or selector tokens.
        from_json_dir: Existing JSON directory to summarize instead of solving.
        recursive: Whether JSON loading should recurse into subdirectories.
        one_supercell: Whether bruteforce should solve only one supercell.
        memory_limit_gb: Frontier-DP value-only memory limit in gigabytes.
        bruteforce_md: Optional bruteforce markdown used for note comparison.
    """

    solver: SolverName
    json_out_dir: pathlib.Path
    markdown_out: pathlib.Path
    target_sizes: list[int]
    tilings: list[str]
    from_json_dir: pathlib.Path | None
    recursive: bool
    one_supercell: bool
    memory_limit_gb: float
    bruteforce_md: pathlib.Path


@dataclasses.dataclass(frozen=True, slots=True)
class CertificationWorkflowResult:
    """Artifacts and records produced by a certification workflow.

    Attributes:
        records: Certification records loaded into the final markdown report.
        json_out_dir: Directory used for JSON output or input.
        markdown_out: Markdown report path.
    """

    records: list[CertificationRecord]
    json_out_dir: pathlib.Path
    markdown_out: pathlib.Path


def _emit(stream: TextIO | None, message: str = "") -> None:
    """Write one progress line to *stream*; ``None`` suppresses output."""
    if stream is not None:
        print(message, file=stream, flush=True)


# ---------------------------------------------------------------------------
# Bruteforce backend
# ---------------------------------------------------------------------------


def _bruteforce_case(
    *,
    tiling: base.Tiling,
    tiling_key: str,
    tiling_display: str,
    target_n: int,
    out_dir: pathlib.Path,
    one_supercell: bool,
    stream: TextIO | None,
) -> list[CertificationRecord]:
    """Solve one case exhaustively via fixed-composition brute force.

    Builds the graph for every valid supercell at *target_n* and solves each
    one by exhaustive fixed-composition enumeration. Results are written to
    *out_dir* immediately after solving.

    Args:
        tiling: Tiling instance.
        tiling_key: Registry key.
        tiling_display: Human-readable label.
        target_n: Requested number of nodes.
        out_dir: Directory for per-case JSON output.
        one_supercell: If ``True``, solve only the squarest valid supercell.
        stream: Optional progress output stream.

    Returns:
        Certification records (one per supercell, or a single skip record).
    """
    if one_supercell:
        m = base.min_valid_supercell_matrix(tiling, target_n)
        matrices = [] if m is None else [m]
    else:
        matrices = base.valid_supercell_matrices(tiling, target_n)

    if not matrices:
        skip = make_skip_record(
            tiling_key=tiling_key,
            tiling_display=tiling_display,
            solver="fixed_composition_bruteforce",
            target_n=target_n,
            vertex_config=tiling.vertex_config,
        )
        write_case_json(out_dir=out_dir, record=skip, generated_by=_GENERATED_BY)
        return [skip]

    records: list[CertificationRecord] = []
    for matrix in matrices:
        graph = tiling.graph(matrix)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        target_composition = (n_nodes - (n_nodes // 2), n_nodes // 2)
        edges = [
            (int(u), int(v), float(data.get("weight", 1.0)))
            for u, v, data in graph.edges(data=True)
        ]
        total_states = math.comb(n_nodes, target_composition[1])
        supercell_str = format_supercell(matrix)
        _emit(stream, f"    supercell={supercell_str} total_states={total_states:,}")

        t0 = time.perf_counter()
        labels, cut_value, metadata = solve_fixed_composition_parallel(
            edges=edges,
            graph=graph,
            target_composition=target_composition,
            stream=stream,
        )
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        optimal_labels = metadata.get("optimal_labels", [labels])
        if not isinstance(optimal_labels, list):
            raise TypeError("optimal_labels metadata must be a list.")

        rec = build_bruteforce_record(
            tiling_key=tiling_key,
            tiling_display=tiling_display,
            vertex_config=tiling.vertex_config,
            target_n=target_n,
            realized_n=n_nodes,
            supercell=supercell_str,
            n_edges=n_edges,
            cut_value=float(cut_value),
            runtime_ms=runtime_ms,
            target_composition=target_composition,
            optimal_labels=optimal_labels,
            total_states=total_states,
            solver_metadata=metadata,
        )
        write_case_json(out_dir=out_dir, record=rec, generated_by=_GENERATED_BY)
        records.append(rec)
    return records


def _run_bruteforce(
    *,
    json_out_dir: pathlib.Path,
    markdown_out: pathlib.Path,
    target_sizes: list[int],
    selected_tilings: list[tuple[str, str]],
    one_supercell: bool,
    recursive: bool,
    stream: TextIO | None,
) -> list[CertificationRecord]:
    """Run the bruteforce backend; return the records in the markdown report."""
    for tiling_key, display in selected_tilings:
        tiling = registry.get(tiling_key)
        _emit(stream, f"\n[{display}] n_v={tiling.n_vertices}")
        for target_n in target_sizes:
            case_records = _bruteforce_case(
                tiling=tiling,
                tiling_key=tiling_key,
                tiling_display=display,
                target_n=target_n,
                out_dir=json_out_dir,
                one_supercell=one_supercell,
                stream=stream,
            )
            _print_case_summary(case_records, target_n, stream)

    _emit(stream, f"\nCase JSON files written to {json_out_dir}")
    return _write_generic_markdown(
        json_out_dir=json_out_dir,
        markdown_out=markdown_out,
        recursive=recursive,
        title="Brute-Force MAX-CUT Certification",
        description=(
            "Fixed-composition MAX-CUT values in this report are independently"
            " certified via a brute-force exact enumerator."
        ),
        extra_columns=_BRUTEFORCE_EXTRA_COLUMNS,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Bruteforce-lookup helpers (used by frontier-dp-value note column)
# ---------------------------------------------------------------------------

# Matches "### 3⁶ (triangular)" style headings in bruteforce-certification.md.
_BF_SECTION_RE: re.Pattern[str] = re.compile(r"^###\s+(.+)$")
# Matches a data row: | n | supercell | k* | ... |
_BF_ROW_RE: re.Pattern[str] = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|.*")


def _load_bruteforce_lookup(
    bf_md: pathlib.Path,
) -> dict[tuple[str, str, str], float]:
    """Parse bruteforce-certification.md into a lookup table.

    Reads per-tiling sections and their data rows to build a map from
    ``(tiling_display, str(n), supercell)`` to the certified k* value.

    Returns:
        Lookup dict; empty if the file does not exist or has no data rows.
    """
    if not bf_md.exists():
        return {}

    lookup: dict[tuple[str, str, str], float] = {}
    current_section = ""
    for raw_line in bf_md.read_text(encoding="utf-8").splitlines():
        section_match = _BF_SECTION_RE.match(raw_line)
        if section_match:
            current_section = section_match.group(1).strip()
            continue
        if not current_section:
            continue
        row_match = _BF_ROW_RE.match(raw_line)
        if not row_match:
            continue
        cells = [c.strip() for c in raw_line.split("|")[1:-1]]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        n_str = cells[0]
        supercell_str = re.sub(r"\s+", "", cells[1])
        try:
            k_star = float(cells[2])
        except ValueError:
            continue
        lookup[(current_section, n_str, supercell_str)] = k_star

    return lookup


def _make_bf_note(
    record: CertificationRecord,
    bf_lookup: dict[tuple[str, str, str], float],
) -> str:
    """Return the note string for one record based on brute-force lookup.

    Returns:
        ``"brute-force agrees"``, ``"brute-force mismatch: {val:.1f}"``,
        or ``""`` if no brute-force data exists for this case.
    """
    if record.certified_cut_value is None or record.supercell is None:
        return ""
    n_str = str(record.realized_n or record.target_n)
    bf_val = bf_lookup.get(
        (record.tiling_display, n_str, re.sub(r"\s+", "", record.supercell))
    )
    if bf_val is None:
        return ""
    if math.isclose(record.certified_cut_value, bf_val, abs_tol=1e-9):
        return "brute-force agrees"
    return f"brute-force mismatch: {bf_val:.1f}"


# ---------------------------------------------------------------------------
# Frontier-DP value-only backend
# ---------------------------------------------------------------------------


def _frontier_dp_value_case(
    *,
    tiling: base.Tiling,
    tiling_key: str,
    tiling_display: str,
    target_n: int,
    out_dir: pathlib.Path,
    memory_limit_gb: float,
    stream: TextIO | None,
) -> list[CertificationRecord]:
    """Certify k* for every valid HNF supercell via the frontier-DP value path.

    Args:
        tiling: Tiling instance.
        tiling_key: Registry key.
        tiling_display: Human-readable label.
        target_n: Requested number of nodes.
        out_dir: Directory for per-case JSON output.
        memory_limit_gb: Skip supercells whose estimated peak memory exceeds
            this threshold in GB.
        stream: Optional progress output stream.

    Returns:
        Certification records (one per supercell, or a single skip record).
    """
    matrices = list(base.valid_supercell_matrices(tiling, target_n))
    if not matrices:
        skip = make_skip_record(
            tiling_key=tiling_key,
            tiling_display=tiling_display,
            solver="frontier_dp_value_only",
            target_n=target_n,
            vertex_config=tiling.vertex_config,
        )
        write_case_json(out_dir=out_dir, record=skip, generated_by=_GENERATED_BY)
        return [skip]

    records: list[CertificationRecord] = []
    for matrix in matrices:
        graph = tiling.graph(matrix)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        n_b = n_nodes // 2
        supercell_str = format_supercell(matrix)

        heuristic, k = select_heuristic(graph)
        frontier_sizes = compute_frontier_sizes(graph, heuristic)
        peak_gb = estimate_peak_memory_bytes(frontier_sizes, n_b) / 1e9
        if peak_gb > memory_limit_gb:
            budget_rec = CertificationRecord(
                tiling_key=tiling_key,
                tiling_display=tiling_display,
                solver="frontier_dp_value_only",
                assignment_index=1,
                n_optimal_assignments=0,
                vertex_config=tiling.vertex_config,
                target_n=target_n,
                realized_n=n_nodes,
                supercell=supercell_str,
                n_edges=n_edges,
                certified_cut_value=None,
                certified_cut_efficiency=None,
                runtime_ms=0.0,
                status="budget",
                note=(
                    f"estimated peak {peak_gb:.0f} GB exceeds limit "
                    f"{memory_limit_gb:.0f} GB (k={k}, n={n_nodes})"
                ),
            )
            write_case_json(
                out_dir=out_dir, record=budget_rec, generated_by=_GENERATED_BY
            )
            records.append(budget_rec)
            continue

        t0 = time.perf_counter()
        try:
            cut_value = compute_max_cut_value(graph, n_b=n_b, heuristic=heuristic)
        except MemoryError:
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            oom_rec = CertificationRecord(
                tiling_key=tiling_key,
                tiling_display=tiling_display,
                solver="frontier_dp_value_only",
                assignment_index=1,
                n_optimal_assignments=0,
                vertex_config=tiling.vertex_config,
                target_n=target_n,
                realized_n=n_nodes,
                supercell=supercell_str,
                n_edges=n_edges,
                certified_cut_value=None,
                certified_cut_efficiency=None,
                runtime_ms=runtime_ms,
                status="oom",
                note="MemoryError",
            )
            write_case_json(
                out_dir=out_dir, record=oom_rec, generated_by=_GENERATED_BY
            )
            records.append(oom_rec)
            continue

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        if runtime_ms / 1000.0 > _SLOW_SOLVE_THRESHOLD_S:
            _emit(
                stream,
                f"  [slow] {tiling_display} n={n_nodes} supercell={supercell_str}"
                f" t={runtime_ms / 1000.0:.1f}s",
            )

        cut_efficiency = cut_value / n_edges if n_edges > 0 else 0.0
        if not math.isclose(cut_value, round(cut_value), abs_tol=1e-9):
            raise ValueError(
                f"cut_value={cut_value!r} is not integral for {tiling_key} "
                f"supercell {supercell_str}; edge weights must all be 1"
            )
        frustrated = n_edges - int(round(cut_value))

        rec = CertificationRecord(
            tiling_key=tiling_key,
            tiling_display=tiling_display,
            solver="frontier_dp_value_only",
            assignment_index=1,
            n_optimal_assignments=0,
            vertex_config=tiling.vertex_config,
            target_n=target_n,
            realized_n=n_nodes,
            supercell=supercell_str,
            n_edges=n_edges,
            certified_cut_value=float(cut_value),
            certified_cut_efficiency=cut_efficiency,
            runtime_ms=runtime_ms,
            status="certified",
            note="",
            metadata={"n_b": n_b, "frustrated": frustrated},
        )
        write_case_json(out_dir=out_dir, record=rec, generated_by=_GENERATED_BY)
        records.append(rec)
    return records


def _run_frontier_dp_value(
    *,
    json_out_dir: pathlib.Path,
    markdown_out: pathlib.Path,
    target_sizes: list[int],
    selected_tilings: list[tuple[str, str]],
    memory_limit_gb: float,
    bruteforce_md: pathlib.Path,
    recursive: bool,
    stream: TextIO | None,
) -> list[CertificationRecord]:
    """Run the frontier-DP value-only backend; return the records in the report.

    Args:
        json_out_dir: Directory for JSON output.
        markdown_out: Path for markdown output.
        target_sizes: Target node counts to solve.
        selected_tilings: ``(key, display)`` pairs to solve.
        memory_limit_gb: Memory budget in gigabytes.
        bruteforce_md: Path to brute-force markdown for note comparison.
        recursive: Whether markdown loading should recurse into subdirectories.
        stream: Optional progress output stream.
    """
    for tiling_key, display in selected_tilings:
        tiling = registry.get(tiling_key)
        _emit(stream, f"\n[{display}] n_v={tiling.n_vertices}")
        for target_n in target_sizes:
            case_records = _frontier_dp_value_case(
                tiling=tiling,
                tiling_key=tiling_key,
                tiling_display=display,
                target_n=target_n,
                out_dir=json_out_dir,
                memory_limit_gb=memory_limit_gb,
                stream=stream,
            )
            _print_case_summary(case_records, target_n, stream)

    _emit(stream, f"\nCase JSON files written to {json_out_dir}")
    bf_lookup = _load_bruteforce_lookup(bruteforce_md)
    if bf_lookup:
        _emit(
            stream,
            f"Loaded {len(bf_lookup)} brute-force entries from {bruteforce_md}",
        )
    return _write_generic_markdown(
        json_out_dir=json_out_dir,
        markdown_out=markdown_out,
        recursive=recursive,
        title="Frontier-DP Value-Only MAX-CUT Certification",
        description=(
            "Exact fixed-composition MAX-CUT values certified by "
            "compute_max_cut_value (frontier dynamic programming, value-only path)."
        ),
        extra_columns=[
            ("note", lambda r: _make_bf_note(r, bf_lookup)),
        ],
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Annealing backend
# ---------------------------------------------------------------------------


def _bipartite_from_graph(g: nx.Graph[int]) -> assignment.AssignmentResult:
    """Return an exact 2-colouring of a bipartite graph.

    ``BipartiteAssigner.assign()`` requires a diagonal supercell tuple; this
    helper covers the shear-matrix case by reading the 2-colouring directly
    from the already-built graph.

    Returns:
        An :class:`~archimono.assignment.base.AssignmentResult` with
        ``cut_value`` equal to the total edge count.
    """
    coloring: dict[int, int] = nx_bipartite.color(g)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    labels = tuple(coloring[i] for i in range(n))
    return assignment.AssignmentResult(
        labels=labels,
        objective_value=float(m),
        cut_value=float(m),
        n_frustrated=0,
        solver="bipartite_exact",
        metadata={"target_composition": (n - sum(labels), sum(labels))},
    )


def _annealing_case(
    *,
    tiling: base.Tiling,
    tiling_key: str,
    tiling_display: str,
    target_n: int,
    out_dir: pathlib.Path,
) -> CertificationRecord:
    """Solve one case via AnnealingSolver (or exact 2-colouring for bipartite).

    Uses the squarest valid supercell at *target_n*.
    """
    sc = base.min_valid_supercell_matrix(tiling, target_n)
    if sc is None:
        skip = make_skip_record(
            tiling_key=tiling_key,
            tiling_display=tiling_display,
            solver="annealing",
            target_n=target_n,
            vertex_config=tiling.vertex_config,
        )
        write_case_json(out_dir=out_dir, record=skip, generated_by=_GENERATED_BY)
        return skip

    graph = tiling.graph(sc)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    supercell_str = format_supercell(sc)

    t0 = time.perf_counter()
    is_diagonal = int(sc[0, 1]) == 0 and int(sc[1, 0]) == 0
    if tiling.is_bipartite and is_diagonal:
        result = BipartiteAssigner.assign(
            tiling, n_b=n_nodes // 2, supercell=(int(sc[0, 0]), int(sc[1, 1]))
        )
        solver_name = "bipartite_exact"
    elif tiling.is_bipartite:
        result = _bipartite_from_graph(graph)
        solver_name = "bipartite_exact"
    else:
        result = _ANNEALING_SOLVER.solve(graph, ["A", "B"], n_b=n_nodes // 2)
        solver_name = "annealing"
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    cut_value = result.cut_value if result.cut_value is not None else 0.0
    cut_efficiency = cut_value / n_edges if n_edges > 0 else 0.0
    is_exact = solver_name == "bipartite_exact"
    case_status = "certified" if is_exact else "heuristic"

    rec = CertificationRecord(
        tiling_key=tiling_key,
        tiling_display=tiling_display,
        solver=solver_name,
        assignment_index=1,
        n_optimal_assignments=1,
        vertex_config=tiling.vertex_config,
        target_n=target_n,
        realized_n=n_nodes,
        supercell=supercell_str,
        n_edges=n_edges,
        certified_cut_value=float(cut_value),
        certified_cut_efficiency=cut_efficiency,
        runtime_ms=runtime_ms,
        status=case_status,
        note="",
        metadata={
            "n_frustrated": result.n_frustrated,
            "target_composition": (n_nodes - n_nodes // 2, n_nodes // 2),
            "is_exact": is_exact,
        },
    )
    write_case_json(out_dir=out_dir, record=rec, generated_by=_GENERATED_BY)
    return rec


def _run_annealing(
    *,
    json_out_dir: pathlib.Path,
    markdown_out: pathlib.Path,
    target_sizes: list[int],
    selected_tilings: list[tuple[str, str]],
    recursive: bool,
    stream: TextIO | None,
) -> list[CertificationRecord]:
    """Run the annealing backend; return the records in the markdown report."""
    for tiling_key, display in selected_tilings:
        tiling = registry.get(tiling_key)
        _emit(
            stream,
            f"\n[{display}] n_v={tiling.n_vertices} bipartite={tiling.is_bipartite}",
        )
        for target_n in target_sizes:
            rec = _annealing_case(
                tiling=tiling,
                tiling_key=tiling_key,
                tiling_display=display,
                target_n=target_n,
                out_dir=json_out_dir,
            )
            k_val = rec.certified_cut_value
            k_star: object = "-" if k_val is None else k_val
            _emit(
                stream,
                f"  n={rec.realized_n or target_n:>3} "
                f"status={rec.status:<9} "
                f"k*={k_star!s:>6} "
                f"runtime_ms={rec.runtime_ms:>8.1f}",
            )

    _emit(stream, f"\nCase JSON files written to {json_out_dir}")
    return _write_generic_markdown(
        json_out_dir=json_out_dir,
        markdown_out=markdown_out,
        recursive=recursive,
        title="Annealing Stress-Test Results",
        description=(
            "Cut efficiency k*/|E| for all selected Archimedean tilings. "
            "Bipartite tilings use exact 2-colouring; frustrated tilings use "
            "AnnealingSolver(n_restarts=30, seed=42)."
        ),
        stream=stream,
        extra_columns=[
            (
                "method",
                lambda r: "-"
                if r.status == "skip"
                else ("exact" if r.solver == "bipartite_exact" else "heuristic"),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Extra report columns shared by the bruteforce run and --from-json-dir paths.
_BRUTEFORCE_EXTRA_COLUMNS: list[tuple[str, Any]] = [
    (
        "n_states_evaluated",
        lambda r: str(r.metadata.get("n_states_evaluated", "-")),
    ),
    (
        "n_optimal_assignments",
        lambda r: str(
            r.metadata.get("n_optimal_assignments", r.n_optimal_assignments)
        ),
    ),
]


def _print_case_summary(
    case_records: list[CertificationRecord],
    target_n: int,
    stream: TextIO | None,
) -> None:
    """Print a one-line summary for a solved case."""
    first = case_records[0]
    total_ms = sum(r.runtime_ms for r in case_records)
    n_supercells = len({r.supercell for r in case_records})
    first_val = first.certified_cut_value
    k_star: object = "-" if first_val is None else first_val
    _emit(
        stream,
        f"  n={first.realized_n or target_n:>3} "
        f"status={first.status:<9} "
        f"k*={k_star!s:>6} "
        f"runtime_ms={total_ms:>8.1f} "
        f"supercells={n_supercells}",
    )


def _write_generic_markdown(
    *,
    json_out_dir: pathlib.Path,
    markdown_out: pathlib.Path,
    recursive: bool,
    title: str,
    description: str,
    extra_columns: (
        list[tuple[str, Any]]
        | None
    ) = None,
    stream: TextIO | None = None,
) -> list[CertificationRecord]:
    """Load records and write a markdown summary.

    Args:
        json_out_dir: Directory containing per-case JSON files.
        markdown_out: Destination path for the markdown file.
        recursive: Whether to recurse into subdirectories.
        title: Report title.
        description: One-sentence description for the report header.
        extra_columns: Optional extra columns forwarded to
            :func:`~archimono.certification.write_markdown_summary`.
        stream: Optional progress output stream.
    """
    if recursive:
        records = load_case_json_tree(json_out_dir)
    else:
        records = load_case_json_dir(json_out_dir)
    _emit(stream, f"Loaded {len(records)} records from {json_out_dir}")
    write_markdown_summary(
        markdown_out,
        records,
        title=title,
        description=description,
        extra_columns=extra_columns,
    )
    _emit(stream, f"Markdown written to {markdown_out}")
    return records


def run_certification_workflow(
    config: CertificationWorkflowConfig,
    *,
    stream: TextIO | None = None,
) -> CertificationWorkflowResult:
    """Run a solver-certification workflow.

    Raises:
        ValueError: If tiling selectors are invalid or the solver is unknown.
        KeyError: If a selected tiling is not registered.
        FileNotFoundError: If summary input references missing sidecar data.
    """
    selected_tilings = resolve_tilings(config.tilings, catalogue=TILINGS)

    if config.from_json_dir is not None:
        extra_columns = None
        if config.solver == "bruteforce":
            extra_columns = _BRUTEFORCE_EXTRA_COLUMNS
        try:
            display_from_dir = config.from_json_dir.resolve().relative_to(
                _PROJECT_ROOT
            )
        except ValueError:
            display_from_dir = config.from_json_dir
        records = _write_generic_markdown(
            json_out_dir=config.from_json_dir,
            markdown_out=config.markdown_out,
            recursive=config.recursive,
            title=f"{config.solver.title()} MAX-CUT Certification",
            description=f"Results loaded from `{display_from_dir.as_posix()}`.",
            extra_columns=extra_columns,
            stream=stream,
        )
        return CertificationWorkflowResult(
            records=records,
            json_out_dir=config.from_json_dir,
            markdown_out=config.markdown_out,
        )

    if config.solver == "bruteforce":
        records = _run_bruteforce(
            json_out_dir=config.json_out_dir,
            markdown_out=config.markdown_out,
            target_sizes=config.target_sizes,
            selected_tilings=selected_tilings,
            one_supercell=config.one_supercell,
            recursive=config.recursive,
            stream=stream,
        )
    elif config.solver == "frontier-dp-value":
        records = _run_frontier_dp_value(
            json_out_dir=config.json_out_dir,
            markdown_out=config.markdown_out,
            target_sizes=config.target_sizes,
            selected_tilings=selected_tilings,
            memory_limit_gb=config.memory_limit_gb,
            bruteforce_md=config.bruteforce_md,
            recursive=config.recursive,
            stream=stream,
        )
    elif config.solver == "annealing":
        records = _run_annealing(
            json_out_dir=config.json_out_dir,
            markdown_out=config.markdown_out,
            target_sizes=config.target_sizes,
            selected_tilings=selected_tilings,
            recursive=config.recursive,
            stream=stream,
        )
    else:
        raise ValueError(f"unknown solver: {config.solver!r}")

    return CertificationWorkflowResult(
        records=records,
        json_out_dir=config.json_out_dir,
        markdown_out=config.markdown_out,
    )
