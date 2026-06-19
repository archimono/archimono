"""Unified certification script for all MAX-CUT solver backends.

Dispatches to one of four solver backends via ``--solver``:

- ``bruteforce``        — exhaustive fixed-composition enumeration
- ``frontier-dp``       — frontier DP with full assignment output
- ``frontier-dp-value`` — frontier DP value-only path (faster, no labels)
- ``annealing``         — simulated annealing (bipartite tilings use exact
                          2-colouring via BipartiteAssigner)

For each solver the script iterates over the selected tilings and target sizes,
writes one JSON file per solved case to ``--json-out-dir``, and writes a
markdown summary to ``--markdown-out``.

Usage examples::

    python scripts/certify.py --solver bruteforce --tilings hexagonal --target-sizes 12
    python scripts/certify.py --solver annealing --max-n 60
    python scripts/certify.py --solver frontier-dp-value --from-json-dir
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import time
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.algorithms import bipartite as nx_bipartite

from archimono import assignment
from archimono.assignment import (
    AnnealingSolver,
    BipartiteAssigner,
    compute_max_cut_value,
    estimate_peak_memory_bytes,
    select_heuristic,
)
from archimono.certification import (
    TILINGS,
    CertificationRecord,
    format_supercell,
    load_case_json_dir,
    load_case_json_tree,
    make_skip_record,
    resolve_target_sizes,
    resolve_tilings,
    write_case_json,
    write_markdown_summary,
)
from archimono.tilings import base, registry

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_SIZES_BRUTEFORCE: list[int] = [12, 24, 36]
_DEFAULT_TARGET_SIZES_FRONTIER_DP: list[int] = [12, 24, 36]
_DEFAULT_TARGET_SIZES_FRONTIER_DP_VALUE: list[int] = [12, 24, 36, 48, 60]
_DEFAULT_TARGET_SIZES_ANNEALING: list[int] = [12, 24, 36, 48, 60]

_DEFAULT_MAX_N_BRUTEFORCE: int = 40
_DEFAULT_MAX_N_FRONTIER_DP: int = 40
_DEFAULT_MAX_N_FRONTIER_DP_VALUE: int = 60
_DEFAULT_MAX_N_ANNEALING: int = 60

_SLOW_SOLVE_THRESHOLD_S: float = 300.0

# Above this node count the frontier-dp backend does NOT enumerate the full
# optimal set (it only certifies the value). The number of optimal assignments
# explodes combinatorially — e.g. kagome n=36 has ~2.1e7 optima (~190 s) — so
# enumeration is gated to small n; larger cases report n_optimal_assignments=0
# with metadata["optima_enumerated"] = False. n=24 stays fast (<=~64k optima,
# <1 s).
_FRONTIER_DP_ENUMERATE_MAX_N: int = 24

_ANNEALING_SOLVER = AnnealingSolver(n_restarts=30, seed=42)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the unified certification script.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified MAX-CUT certification script. "
            "Dispatches to the selected solver backend."
        )
    )
    parser.add_argument(
        "--solver",
        choices=["bruteforce", "frontier-dp", "frontier-dp-value", "annealing"],
        required=True,
        help="Solver backend to use.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help=(
            "Maximum target size to consider when --target-sizes is omitted. "
            "Defaults to 40 for bruteforce/frontier-dp and 60 for "
            "frontier-dp-value/annealing."
        ),
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit target sizes. Overrides the per-solver defaults. "
            "Passing the flag with no values selects no sizes (runs nothing)."
        ),
    )
    parser.add_argument(
        "--tilings",
        nargs="*",
        default=None,
        help=(
            "Tiling keys to solve. Accepts individual registry keys, 'all', or "
            "'frustrated' (the 7 non-bipartite tilings). "
            "Defaults to 'all' for bruteforce/frontier-dp-value/annealing and "
            "'frustrated' for frontier-dp."
        ),
    )
    parser.add_argument(
        "--json-out-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Directory for per-case JSON output. "
            "Defaults vary by solver (see module docstring)."
        ),
    )
    parser.add_argument(
        "--markdown-out",
        type=pathlib.Path,
        default=None,
        help=(
            "Path for markdown summary output. Defaults to a per-solver "
            "tmp/ path so a partial run cannot overwrite a committed report; "
            "pass an explicit path (e.g. docs/reference/certified-max-cut.md) "
            "to (re)generate a committed report."
        ),
    )
    parser.add_argument(
        "--from-json-dir",
        type=pathlib.Path,
        nargs="?",
        const=True,  # sentinel: use the solver's default json-out-dir
        default=None,
        help=(
            "Skip solving and rebuild markdown from existing JSON files. "
            "Pass a directory path or omit the path argument to use the "
            "solver's default json-out-dir."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --from-json-dir is set, recurse into subdirectories.",
    )
    # bruteforce-only flags
    parser.add_argument(
        "--one-supercell",
        action="store_true",
        help="[bruteforce only] Solve only the single squarest supercell per case.",
    )
    # frontier-dp-value-only flags
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=60.0,
        help=(
            "[frontier-dp-value only] Skip supercells whose estimated peak "
            "memory exceeds this limit in GB. Default: 60."
        ),
    )
    parser.add_argument(
        "--bruteforce-md",
        type=pathlib.Path,
        default=pathlib.Path("docs/draft/bruteforce-certification.md"),
        help=(
            "[frontier-dp-value only] Path to bruteforce-certification.md "
            "for the note column. Ignored if the file does not exist."
        ),
    )
    return parser.parse_args()


def _default_json_out_dir(solver: str) -> pathlib.Path:
    """Return the default JSON output directory for *solver*.

    Args:
        solver: One of ``bruteforce``, ``frontier-dp``, ``frontier-dp-value``,
            or ``annealing``.

    Returns:
        Default output directory path.
    """
    return {
        "bruteforce": pathlib.Path("tmp/bruteforce-certification"),
        "frontier-dp": pathlib.Path("tmp/frontier-dp-exact"),
        "frontier-dp-value": pathlib.Path("tmp/certify-max-cut"),
        "annealing": pathlib.Path("tmp/annealing-stress"),
    }[solver]


def _default_markdown_out(solver: str) -> pathlib.Path:
    """Return the default markdown output path for *solver*.

    Defaults write under ``tmp/`` (not into the committed docs tree) so a
    partial or smoke run cannot silently overwrite a committed report. To
    (re)generate a committed report, pass ``--markdown-out`` explicitly, e.g.
    ``--markdown-out docs/reference/certified-max-cut.md`` for
    ``frontier-dp-value``.

    Args:
        solver: One of ``bruteforce``, ``frontier-dp``, ``frontier-dp-value``,
            or ``annealing``.

    Returns:
        Default markdown output path under ``tmp/``.
    """
    return {
        "bruteforce": pathlib.Path("tmp/bruteforce-certification.md"),
        "frontier-dp": pathlib.Path("tmp/frontier-dp-exact-results.md"),
        "frontier-dp-value": pathlib.Path("tmp/certified-max-cut.md"),
        "annealing": pathlib.Path("tmp/stress-test-results.md"),
    }[solver]


def _default_tilings(solver: str) -> list[str]:
    """Return the default tiling token list for *solver*.

    Args:
        solver: One of ``bruteforce``, ``frontier-dp``, ``frontier-dp-value``,
            or ``annealing``.

    Returns:
        Default tiling token list.
    """
    if solver == "frontier-dp":
        return ["frustrated"]
    return ["all"]


def _default_max_n(solver: str) -> int:
    """Return the default ``--max-n`` value for *solver*.

    Args:
        solver: One of ``bruteforce``, ``frontier-dp``, ``frontier-dp-value``,
            or ``annealing``.

    Returns:
        Default maximum target size.
    """
    return {
        "bruteforce": _DEFAULT_MAX_N_BRUTEFORCE,
        "frontier-dp": _DEFAULT_MAX_N_FRONTIER_DP,
        "frontier-dp-value": _DEFAULT_MAX_N_FRONTIER_DP_VALUE,
        "annealing": _DEFAULT_MAX_N_ANNEALING,
    }[solver]


def _default_target_sizes(solver: str) -> list[int]:
    """Return the default target size list for *solver*.

    Args:
        solver: One of ``bruteforce``, ``frontier-dp``, ``frontier-dp-value``,
            or ``annealing``.

    Returns:
        Default target sizes.
    """
    return {
        "bruteforce": _DEFAULT_TARGET_SIZES_BRUTEFORCE,
        "frontier-dp": _DEFAULT_TARGET_SIZES_FRONTIER_DP,
        "frontier-dp-value": _DEFAULT_TARGET_SIZES_FRONTIER_DP_VALUE,
        "annealing": _DEFAULT_TARGET_SIZES_ANNEALING,
    }[solver]


# ---------------------------------------------------------------------------
# Bruteforce backend
# ---------------------------------------------------------------------------


def _diagonal_supercell_tuple(
    matrix: npt.NDArray[np.intp],
) -> tuple[int, int] | None:
    """Convert a diagonal supercell matrix into an ``(a, b)`` tuple.

    Args:
        matrix: 2×2 integer array.

    Returns:
        ``(a, b)`` for diagonal matrices, ``None`` for shear matrices.
    """
    if int(matrix[0, 1]) != 0 or int(matrix[1, 0]) != 0:
        return None
    return int(matrix[0, 0]), int(matrix[1, 1])


def _bruteforce_case(
    *,
    tiling: base.Tiling,
    tiling_key: str,
    tiling_display: str,
    target_n: int,
    out_dir: pathlib.Path,
    one_supercell: bool,
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
        write_case_json(out_dir=out_dir, record=skip, generated_by="scripts/certify.py")
        return [skip]

    records: list[CertificationRecord] = []
    for matrix in matrices:
        graph = tiling.graph(matrix)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        assignment._validation.validate_binary_species_problem(graph, ["A", "B"])  # noqa: SLF001
        target_composition = (n_nodes - (n_nodes // 2), n_nodes // 2)
        edges = [
            (int(u), int(v), float(data.get("weight", 1.0)))
            for u, v, data in graph.edges(data=True)
        ]
        total_states = math.comb(n_nodes, target_composition[1])
        supercell_str = format_supercell(matrix)
        print(
            f"    supercell={supercell_str} total_states={total_states:,}",
            flush=True,
        )

        t0 = time.perf_counter()
        labels, cut_value, metadata = assignment._bruteforce.solve_fixed_composition(  # noqa: SLF001
            edges=edges,
            graph=graph,
            target_composition=target_composition,
        )
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        cut_efficiency = cut_value / n_edges if n_edges > 0 else 0.0

        # Cross-check with BruteforceSolver on the (2×2) diagonal supercell.
        exact_planar_cut_value: float | None = None
        diag = _diagonal_supercell_tuple(matrix)
        if diag == (2, 2):
            exact_graph = tiling.graph(diag)
            try:
                ep_result = assignment.BruteforceSolver().solve(
                    exact_graph, ["A", "B"], n_b=n_nodes // 2
                )
                exact_planar_cut_value = (
                    None
                    if ep_result.cut_value is None
                    else float(ep_result.cut_value)
                )
            except ValueError:
                exact_planar_cut_value = None

        optimal_labels = metadata.get("optimal_labels", [labels])
        if not isinstance(optimal_labels, list):
            raise TypeError("optimal_labels metadata must be a list.")

        rec = CertificationRecord(
            tiling_key=tiling_key,
            tiling_display=tiling_display,
            solver="fixed_composition_bruteforce",
            assignment_index=1,
            n_optimal_assignments=len(optimal_labels),
            vertex_config=tiling.vertex_config,
            target_n=target_n,
            realized_n=n_nodes,
            supercell=supercell_str,
            n_edges=n_edges,
            certified_cut_value=float(cut_value),
            certified_cut_efficiency=cut_efficiency,
            runtime_ms=runtime_ms,
            status="certified",
            exact_planar_cut_value=exact_planar_cut_value,
            note="",
            metadata={
                "species": ("A", "B"),
                "target_composition": target_composition,
                "all_labels": [tuple(lbl) for lbl in optimal_labels],
                "n_states_total": total_states,
                **{k: v for k, v in metadata.items() if k != "optimal_labels"},
            },
        )
        write_case_json(out_dir=out_dir, record=rec, generated_by="scripts/certify.py")
        records.append(rec)
    return records


def _run_bruteforce(
    args: argparse.Namespace,
    json_out_dir: pathlib.Path,
    markdown_out: pathlib.Path,
    target_sizes: list[int],
    selected_tilings: list[tuple[str, str]],
) -> None:
    """Run the bruteforce solver backend.

    Args:
        args: Parsed CLI arguments.
        json_out_dir: Directory for JSON output.
        markdown_out: Path for markdown output.
        target_sizes: Target node counts to solve.
        selected_tilings: ``(key, display)`` pairs to solve.
    """
    registry._ensure_loaded()
    for tiling_key, display in selected_tilings:
        tiling = registry.get(tiling_key)
        print(f"\n[{display}] n_v={tiling.n_vertices}", flush=True)
        for target_n in target_sizes:
            case_records = _bruteforce_case(
                tiling=tiling,
                tiling_key=tiling_key,
                tiling_display=display,
                target_n=target_n,
                out_dir=json_out_dir,
                one_supercell=args.one_supercell,
            )
            _print_case_summary(case_records, target_n)

    print(f"\nCase JSON files written to {json_out_dir}", flush=True)
    _write_generic_markdown(
        json_out_dir=json_out_dir,
        markdown_out=markdown_out,
        recursive=args.recursive,
        title="Brute-Force MAX-CUT Certification",
        description=(
            "Fixed-composition MAX-CUT values in this report are independently"
            " certified via a brute-force exact enumerator."
        ),
        extra_columns=[
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
        ],
    )


# ---------------------------------------------------------------------------
# Frontier-DP backend
# ---------------------------------------------------------------------------


def _frontier_dp_case(
    *,
    tiling: base.Tiling,
    tiling_key: str,
    tiling_display: str,
    target_n: int,
    out_dir: pathlib.Path,
) -> list[CertificationRecord]:
    """Solve one case via FrontierExactSolver.

    Args:
        tiling: Tiling instance.
        tiling_key: Registry key.
        tiling_display: Human-readable label.
        target_n: Requested number of nodes.
        out_dir: Directory for per-case JSON output.

    Returns:
        Certification records (one per supercell, or a single skip record).
    """
    matrices = base.valid_supercell_matrices(tiling, target_n)
    if not matrices:
        skip = make_skip_record(
            tiling_key=tiling_key,
            tiling_display=tiling_display,
            solver="frontier_dp",
            target_n=target_n,
            vertex_config=tiling.vertex_config,
        )
        write_case_json(out_dir=out_dir, record=skip, generated_by="scripts/certify.py")
        return [skip]

    solver = assignment.FrontierExactSolver()
    records: list[CertificationRecord] = []
    for matrix in matrices:
        graph = tiling.graph(matrix)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        supercell_str = format_supercell(matrix)

        # Enumerate all optima only when the instance is small enough that the
        # optimal-set materialization stays bounded; otherwise report the count
        # as "not enumerated" (optima_enumerated=False) rather than a misleading
        # constant 1.
        do_enumerate = n_nodes <= _FRONTIER_DP_ENUMERATE_MAX_N
        t0 = time.perf_counter()
        result = solver.solve(
            graph, ["A", "B"], n_b=n_nodes // 2, enumerate_all=do_enumerate
        )
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        cut_value = (
            None if result.cut_value is None else float(result.cut_value)
        )
        cut_efficiency = (
            None if cut_value is None or n_edges == 0 else cut_value / n_edges
        )
        all_opt = result.metadata.get("all_labels")
        n_optimal = len(all_opt) if do_enumerate and all_opt is not None else 0

        rec = CertificationRecord(
            tiling_key=tiling_key,
            tiling_display=tiling_display,
            solver="frontier_dp",
            assignment_index=1,
            n_optimal_assignments=n_optimal,
            vertex_config=tiling.vertex_config,
            target_n=target_n,
            realized_n=n_nodes,
            supercell=supercell_str,
            n_edges=n_edges,
            certified_cut_value=cut_value,
            certified_cut_efficiency=cut_efficiency,
            runtime_ms=runtime_ms,
            status="certified",
            exact_planar_cut_value=None,
            note="",
            # Strip "all_labels": at n<=_FRONTIER_DP_ENUMERATE_MAX_N the optimal
            # set is enumerated only to COUNT it (n_optimal_assignments); storing
            # every optimum would write a huge sidecar (e.g. ~64k labelings at
            # n=24). The single representative is kept under "labels".
            metadata={
                **{
                    k: v for k, v in result.metadata.items() if k != "all_labels"
                },
                "labels": result.labels,
                "optima_enumerated": do_enumerate,
            },
        )
        write_case_json(out_dir=out_dir, record=rec, generated_by="scripts/certify.py")
        records.append(rec)
    return records


def _run_frontier_dp(
    args: argparse.Namespace,
    json_out_dir: pathlib.Path,
    markdown_out: pathlib.Path,
    target_sizes: list[int],
    selected_tilings: list[tuple[str, str]],
) -> None:
    """Run the frontier-DP solver backend.

    Args:
        args: Parsed CLI arguments.
        json_out_dir: Directory for JSON output.
        markdown_out: Path for markdown output.
        target_sizes: Target node counts to solve.
        selected_tilings: ``(key, display)`` pairs to solve.
    """
    registry._ensure_loaded()
    for tiling_key, display in selected_tilings:
        tiling = registry.get(tiling_key)
        print(f"\n[{display}] n_v={tiling.n_vertices}", flush=True)
        for target_n in target_sizes:
            case_records = _frontier_dp_case(
                tiling=tiling,
                tiling_key=tiling_key,
                tiling_display=display,
                target_n=target_n,
                out_dir=json_out_dir,
            )
            _print_case_summary(case_records, target_n)

    print(f"\nCase JSON files written to {json_out_dir}", flush=True)
    _write_generic_markdown(
        json_out_dir=json_out_dir,
        markdown_out=markdown_out,
        recursive=args.recursive,
        title="Frontier-DP Exact MAX-CUT Results",
        description=(
            "Fixed-composition MAX-CUT values in this report are solved exactly"
            " by FrontierExactSolver (frontier dynamic programming,"
            " geometric heuristic)."
        ),
        extra_columns=[
            (
                "n_states_explored",
                lambda r: str(r.metadata.get("n_states_explored", "-")),
            ),
            (
                "max_states_retained",
                lambda r: str(r.metadata.get("max_states_retained", "-")),
            ),
            (
                "order_name",
                lambda r: str(r.metadata.get("order_name", "-")),
            ),
        ],
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

    Args:
        bf_md: Path to the bruteforce-certification markdown file.

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

    Args:
        record: Frontier-DP certification record.
        bf_lookup: Lookup table from :func:`_load_bruteforce_lookup`.

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
        write_case_json(out_dir=out_dir, record=skip, generated_by="scripts/certify.py")
        return [skip]

    records: list[CertificationRecord] = []
    for matrix in matrices:
        graph = tiling.graph(matrix)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        n_b = n_nodes // 2
        supercell_str = format_supercell(matrix)

        heuristic, k = select_heuristic(graph)
        peak_gb = estimate_peak_memory_bytes(k, n_nodes) / 1e9
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
                out_dir=out_dir, record=budget_rec, generated_by="scripts/certify.py"
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
                out_dir=out_dir, record=oom_rec, generated_by="scripts/certify.py"
            )
            records.append(oom_rec)
            continue

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        if runtime_ms / 1000.0 > _SLOW_SOLVE_THRESHOLD_S:
            print(
                f"  [slow] {tiling_display} n={n_nodes} supercell={supercell_str}"
                f" t={runtime_ms / 1000.0:.1f}s",
                flush=True,
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
        write_case_json(out_dir=out_dir, record=rec, generated_by="scripts/certify.py")
        records.append(rec)
    return records


def _run_frontier_dp_value(
    args: argparse.Namespace,
    json_out_dir: pathlib.Path,
    markdown_out: pathlib.Path,
    target_sizes: list[int],
    selected_tilings: list[tuple[str, str]],
) -> None:
    """Run the frontier-DP value-only solver backend.

    Args:
        args: Parsed CLI arguments.
        json_out_dir: Directory for JSON output.
        markdown_out: Path for markdown output.
        target_sizes: Target node counts to solve.
        selected_tilings: ``(key, display)`` pairs to solve.
    """
    registry._ensure_loaded()
    for tiling_key, display in selected_tilings:
        tiling = registry.get(tiling_key)
        print(f"\n[{display}] n_v={tiling.n_vertices}", flush=True)
        for target_n in target_sizes:
            case_records = _frontier_dp_value_case(
                tiling=tiling,
                tiling_key=tiling_key,
                tiling_display=display,
                target_n=target_n,
                out_dir=json_out_dir,
                memory_limit_gb=args.memory_limit_gb,
            )
            _print_case_summary(case_records, target_n)

    print(f"\nCase JSON files written to {json_out_dir}", flush=True)
    bf_lookup = _load_bruteforce_lookup(args.bruteforce_md)
    if bf_lookup:
        print(
            f"Loaded {len(bf_lookup)} brute-force entries from {args.bruteforce_md}",
            flush=True,
        )
    _write_generic_markdown(
        json_out_dir=json_out_dir,
        markdown_out=markdown_out,
        recursive=args.recursive,
        title="Frontier-DP Value-Only MAX-CUT Certification",
        description=(
            "Exact fixed-composition MAX-CUT values certified by "
            "compute_max_cut_value (frontier dynamic programming, value-only path)."
        ),
        extra_columns=[
            ("note", lambda r: _make_bf_note(r, bf_lookup)),
        ],
    )


# ---------------------------------------------------------------------------
# Annealing backend
# ---------------------------------------------------------------------------


def _bipartite_from_graph(g: nx.Graph[int]) -> assignment.AssignmentResult:
    """Return an exact 2-colouring of a bipartite graph.

    ``BipartiteAssigner.assign()`` requires a diagonal supercell tuple; this
    helper covers the shear-matrix case by reading the 2-colouring directly
    from the already-built graph.

    Args:
        g: A bipartite NetworkX graph with integer node labels.

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

    Args:
        tiling: Tiling instance.
        tiling_key: Registry key.
        tiling_display: Human-readable label.
        target_n: Requested number of nodes.
        out_dir: Directory for per-case JSON output.

    Returns:
        A single :class:`CertificationRecord`.
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
        write_case_json(out_dir=out_dir, record=skip, generated_by="scripts/certify.py")
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

    # Only the exact 2-colouring is certified optimal; annealing is heuristic
    # and must not masquerade as a proven optimum.
    is_exact = solver_name == "bipartite_exact"
    case_status = "certified" if is_exact else "heuristic"

    cut_value = result.cut_value if result.cut_value is not None else 0.0
    cut_efficiency = cut_value / n_edges if n_edges > 0 else 0.0

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
    write_case_json(out_dir=out_dir, record=rec, generated_by="scripts/certify.py")
    return rec


def _run_annealing(
    args: argparse.Namespace,
    json_out_dir: pathlib.Path,
    markdown_out: pathlib.Path,
    target_sizes: list[int],
    selected_tilings: list[tuple[str, str]],
) -> None:
    """Run the annealing solver backend.

    Args:
        args: Parsed CLI arguments.
        json_out_dir: Directory for JSON output.
        markdown_out: Path for markdown output.
        target_sizes: Target node counts to solve.
        selected_tilings: ``(key, display)`` pairs to solve.
    """
    registry._ensure_loaded()
    for tiling_key, display in selected_tilings:
        tiling = registry.get(tiling_key)
        print(
            f"\n[{display}] n_v={tiling.n_vertices} bipartite={tiling.is_bipartite}",
            flush=True,
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
            print(
                f"  n={rec.realized_n or target_n:>3} "
                f"status={rec.status:<9} "
                f"k*={k_star!s:>6} "
                f"runtime_ms={rec.runtime_ms:>8.1f}",
                flush=True,
            )

    print(f"\nCase JSON files written to {json_out_dir}", flush=True)
    _write_generic_markdown(
        json_out_dir=json_out_dir,
        markdown_out=markdown_out,
        recursive=args.recursive,
        title="Annealing Stress-Test Results",
        description=(
            "Cut efficiency k*/|E| for all selected Archimedean tilings. "
            "Bipartite tilings use exact 2-colouring (certified optimal); "
            "frustrated tilings use AnnealingSolver(n_restarts=30, seed=42), a "
            "heuristic whose k* is the best value found, not a certified "
            "optimum. The 'method' column marks each row exact vs heuristic."
        ),
        extra_columns=[
            (
                "method",
                lambda r: "-"
                if r.status == "skip"
                else ("exact" if r.solver == "bipartite_exact" else "heuristic"),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _print_case_summary(
    case_records: list[CertificationRecord],
    target_n: int,
) -> None:
    """Print a one-line summary for a solved case.

    Args:
        case_records: Records produced for one ``(tiling, target_n)`` case.
        target_n: The target size requested for this case.
    """
    first = case_records[0]
    total_ms = sum(r.runtime_ms for r in case_records)
    n_supercells = len({r.supercell for r in case_records})
    first_val = first.certified_cut_value
    k_star: object = "-" if first_val is None else first_val
    print(
        f"  n={first.realized_n or target_n:>3} "
        f"status={first.status:<9} "
        f"k*={k_star!s:>6} "
        f"runtime_ms={total_ms:>8.1f} "
        f"supercells={n_supercells}",
        flush=True,
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
) -> None:
    """Load records and write a markdown summary.

    Args:
        json_out_dir: Directory containing per-case JSON files.
        markdown_out: Destination path for the markdown file.
        recursive: Whether to recurse into subdirectories.
        title: Report title.
        description: One-sentence description for the report header.
        extra_columns: Optional extra columns forwarded to
            :func:`~archimono.certification.write_markdown_summary`.
    """
    if recursive:
        records = load_case_json_tree(json_out_dir)
    else:
        records = load_case_json_dir(json_out_dir)
    print(f"Loaded {len(records)} records from {json_out_dir}", flush=True)
    write_markdown_summary(
        markdown_out,
        records,
        title=title,
        description=description,
        extra_columns=extra_columns,
    )
    print(f"Markdown written to {markdown_out}", flush=True)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_SOLVER_BACKENDS = {
    "bruteforce": _run_bruteforce,
    "frontier-dp": _run_frontier_dp,
    "frontier-dp-value": _run_frontier_dp_value,
    "annealing": _run_annealing,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the unified certification workflow from the command line."""
    args = _parse_args()
    solver = args.solver

    # Resolve effective defaults.
    json_out_dir = args.json_out_dir or _default_json_out_dir(solver)
    markdown_out = args.markdown_out or _default_markdown_out(solver)
    tiling_keys = args.tilings or _default_tilings(solver)
    max_n = args.max_n if args.max_n is not None else _default_max_n(solver)
    target_sizes = resolve_target_sizes(
        args.target_sizes,
        default=_default_target_sizes(solver),
        max_n=max_n,
    )
    selected_tilings = resolve_tilings(tiling_keys, catalogue=TILINGS)

    # --from-json-dir: rebuild markdown only.
    if args.from_json_dir is not None:
        # If the sentinel True was stored (--from-json-dir with no path arg),
        # fall back to the solver's default dir.
        from_dir = json_out_dir if args.from_json_dir is True else args.from_json_dir
        _write_generic_markdown(
            json_out_dir=from_dir,
            markdown_out=markdown_out,
            recursive=args.recursive,
            title=f"{solver.title()} MAX-CUT Certification",
            description=f"Results loaded from {from_dir}.",
        )
        return

    _SOLVER_BACKENDS[solver](
        args,
        json_out_dir,
        markdown_out,
        target_sizes,
        selected_tilings,
    )


if __name__ == "__main__":
    main()
