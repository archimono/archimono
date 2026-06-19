"""Consolidated benchmarking script for archimono.

Subcommands
-----------
frontier-dp
    Compare FrontierExactSolver / FrontierExactSolverCK / FrontierExactSolverGreedy
    side-by-side across tilings and target sizes.

frontier-order
    Compare vertex-ordering heuristics by maximum frontier width.

numba-jit
    Benchmark the @njit hot-path kernels against their pure-Python equivalents.

sample-runs
    Sample per-restart solver runs across the Archimedean-tiling corpus and
    write results to a SQLite database.

Usage::

    python scripts/benchmark.py frontier-dp --tilings hexagonal --target-sizes 12 24
    python scripts/benchmark.py frontier-order --tilings hexagonal --target-sizes 12
    python scripts/benchmark.py numba-jit
    python scripts/benchmark.py sample-runs --db tmp/runs.db --tilings hexagonal \\
        --target-sizes 12 --n-runs 5
"""

from __future__ import annotations

import argparse
import copy
import math
import pathlib
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt

from archimono import assignment
from archimono.assignment import (
    AnnealingSolver,
    AssignmentSolver,
    GreedySolver,
    compute_frontier_width,
    estimate_peak_memory_bytes,
)
from archimono.benchmarking import (
    SWEEP_CONFIGS,
    SolveRun,
    SolveRunDB,
    SweepConfig,
    insert_run,
    open_db,
)
from archimono.tilings import base, registry
from archimono.tilings.base import Tiling, valid_supercell_matrices

# ---------------------------------------------------------------------------
# Shared tiling catalogue
# ---------------------------------------------------------------------------

_ALL_TILINGS: list[tuple[str, str]] = [
    ("hexagonal", "6³"),
    ("square", "4⁴"),
    ("truncated_square", "4.8²"),
    ("truncated_trihexagonal", "4.6.12"),
    ("triangular", "3⁶"),
    ("kagome", "3.6.3.6"),
    ("truncated_hexagonal", "3.12.12"),
    ("elongated_triangular", "3³.4²"),
    ("rhombitrihexagonal", "3.4.6.4"),
    ("snub_square", "3².4.3.4"),
    ("snub_hexagonal", "3⁴.6"),
]

_FRUSTRATED_TILING_KEYS: frozenset[str] = frozenset({
    "triangular",
    "kagome",
    "truncated_hexagonal",
    "elongated_triangular",
    "rhombitrihexagonal",
    "snub_square",
    "snub_hexagonal",
})

SPECIES: tuple[str, str] = ("A", "B")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_supercell_diag(matrix: npt.NDArray[np.intp]) -> str:
    """Format a 2×2 supercell matrix as a short string.

    Args:
        matrix: 2×2 integer supercell matrix.

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


def _format_supercell_upper(matrix: npt.NDArray[np.intp]) -> str:
    """Format a 2×2 upper-triangular supercell matrix.

    Args:
        matrix: 2×2 integer supercell matrix.

    Returns:
        String like ``(2x3)`` when off-diagonal is zero, or
        ``[a b; 0 d]`` otherwise.
    """
    a, b, d = int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 1])
    if b == 0:
        return f"({a}x{d})"
    return f"[{a} {b}; 0 {d}]"


def _resolve_target_sizes(args: argparse.Namespace, default: list[int]) -> list[int]:
    """Resolve the target-size list from parsed CLI args.

    Args:
        args: Parsed CLI namespace; must have ``target_sizes`` and ``max_n``.
        default: Fallback list used when ``args.target_sizes`` is empty/None.

    Returns:
        Sorted, deduplicated list of positive target sizes.
    """
    if args.target_sizes:
        return sorted({n for n in args.target_sizes if n > 0})
    return [n for n in default if n <= args.max_n]


def _selected_tilings(
    args: argparse.Namespace,
    catalogue: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return the subset of ``catalogue`` selected by ``args.tilings``.

    Args:
        args: Parsed CLI namespace; must have a ``tilings`` attribute.
        catalogue: Full list of ``(key, display)`` tuples.

    Returns:
        Filtered list of ``(key, display)`` tuples.

    Raises:
        ValueError: If any requested tiling key is not in ``catalogue``.
    """
    if args.tilings is None:
        return catalogue
    selected = {name.lower() for name in args.tilings}
    if "frustrated" in selected:
        selected = (selected - {"frustrated"}) | _FRUSTRATED_TILING_KEYS
    matches = [(k, d) for k, d in catalogue if k.lower() in selected]
    unknown = sorted(selected - {k.lower() for k, _ in catalogue})
    if unknown:
        raise ValueError(f"Unknown tilings: {unknown}")
    return matches


# ===========================================================================
# frontier-dp subcommand
# ===========================================================================

@dataclass(slots=True)
class _FrontierDPRow:
    """Per-case row for the frontier-dp benchmark table."""

    tiling_key: str
    tiling_display: str
    supercell: str
    n_nodes: int
    n_edges: int
    cut_v1: float
    cut_ck: float
    cut_gr: float
    agree: str
    runtime_ms_v1: float
    runtime_ms_ck: float
    runtime_ms_gr: float
    n_states_v1: int
    n_states_ck: int
    n_states_gr: int
    order_v1: str


_FDP_DEFAULT_SIZES: list[int] = [12, 24, 36]

_FDP_HEADER = (
    f"{'tiling':<24} {'sup':>8} {'n':>4} {'e':>5} "
    f"{'v1_ms':>8} {'ck_ms':>8} {'gr_ms':>8} "
    f"{'v1_st':>8} {'ck_st':>8} {'gr_st':>8} "
    f"{'v1_order':<22} {'agree':>6}"
)
_FDP_SEP = "-" * len(_FDP_HEADER)


def _run_frontier_solver(
    solver: AssignmentSolver,
    graph: nx.Graph[int],
    n_nodes: int,
) -> tuple[float, float, int, str]:
    """Run one frontier-DP solver variant and return metrics.

    Args:
        solver: The solver instance to run.
        graph: The input graph.
        n_nodes: Number of nodes (used to set n_b = n_nodes // 2).

    Returns:
        Tuple of (cut_value, runtime_ms, n_states_explored, order_name).
    """
    t0 = time.perf_counter()
    result = solver.solve(graph, ["A", "B"], n_b=n_nodes // 2)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    meta = result.metadata
    return (
        float(result.cut_value or 0.0),
        runtime_ms,
        int(meta.get("n_states_explored", 0)),
        str(meta.get("order_name", "?")),
    )


def _render_frontier_dp_markdown(rows: list[_FrontierDPRow]) -> str:
    """Render frontier-dp benchmark rows as a markdown table.

    Args:
        rows: Per-case benchmark result dataclass instances.

    Returns:
        Markdown text with a summary header and a results table.
    """
    n_cases = len(rows)
    mismatches = sum(1 for r in rows if r.agree != "ok")

    def avg_ratio(a: list[float], b: list[float]) -> float:
        """Compute the average ratio a[i]/b[i] for pairs where b[i] > 0."""
        pairs = [(x, y) for x, y in zip(a, b) if y > 0]
        return sum(x / y for x, y in pairs) / len(pairs) if pairs else float("nan")

    ms_v1 = [r.runtime_ms_v1 for r in rows]
    ms_ck = [r.runtime_ms_ck for r in rows]
    ms_gr = [r.runtime_ms_gr for r in rows]
    st_v1 = [float(r.n_states_v1) for r in rows]
    st_ck = [float(r.n_states_ck) for r in rows]
    st_gr = [float(r.n_states_gr) for r in rows]

    lines = [
        "# Frontier DP Benchmark: v1 vs CK vs greedy",
        "",
        "- **v1**: `FrontierExactSolver` — geometric sweeps,"
        " max-frontier-size selection",
        "- **ck**: `FrontierExactSolverCK` — Cuthill-McKee only, no candidate overhead",
        "- **gr**: `FrontierExactSolverGreedy` — greedy min-degree only,"
        " no candidate overhead",
        "",
        "## Summary",
        "",
        f"- Cases: {n_cases}",
        f"- Cut-value mismatches: {mismatches}",
        f"- Avg runtime v1/ck: {avg_ratio(ms_v1, ms_ck):.2f}x",
        f"- Avg runtime v1/gr: {avg_ratio(ms_v1, ms_gr):.2f}x",
        f"- Avg state reduction v1/ck: {avg_ratio(st_v1, st_ck):.2f}x",
        f"- Avg state reduction v1/gr: {avg_ratio(st_v1, st_gr):.2f}x",
        "",
        "## Results",
        "",
        "| tiling | supercell | n | edges"
        " | v1 ms | ck ms | gr ms"
        " | v1 states | ck states | gr states"
        " | v1 order | agree |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.tiling_display} | {r.supercell} | {r.n_nodes} | {r.n_edges}"
            f" | {r.runtime_ms_v1:.1f} | {r.runtime_ms_ck:.1f}"
            f" | {r.runtime_ms_gr:.1f}"
            f" | {r.n_states_v1} | {r.n_states_ck} | {r.n_states_gr}"
            f" | {r.order_v1} | {r.agree} |"
        )
    lines.append("")
    return "\n".join(lines)


def _cmd_frontier_dp(args: argparse.Namespace) -> None:
    """Run the frontier-dp benchmark subcommand.

    Args:
        args: Parsed CLI namespace with target_sizes, max_n, tilings, md_out.
    """
    target_sizes = _resolve_target_sizes(args, _FDP_DEFAULT_SIZES)
    selected = _selected_tilings(args, _ALL_TILINGS)
    registry._ensure_loaded()

    solver_v1 = assignment.FrontierExactSolver()
    solver_ck = assignment.FrontierExactSolverCK()
    solver_gr = assignment.FrontierExactSolverGreedy()

    rows: list[_FrontierDPRow] = []

    print(_FDP_SEP)
    print(_FDP_HEADER)
    print(_FDP_SEP)

    for tiling_key, display in selected:
        tiling = registry.get(tiling_key)
        for target_n in target_sizes:
            matrices = base.valid_supercell_matrices(tiling, target_n)
            if not matrices:
                continue
            for matrix in matrices:
                graph = tiling.graph(matrix)
                n_nodes = graph.number_of_nodes()
                n_edges = graph.number_of_edges()
                sup = _format_supercell_diag(matrix)

                cut_v1, ms_v1, st_v1, order_v1 = _run_frontier_solver(
                    solver_v1, graph, n_nodes
                )
                cut_ck, ms_ck, st_ck, _order_ck = _run_frontier_solver(
                    solver_ck, graph, n_nodes
                )
                cut_gr, ms_gr, st_gr, _order_gr = _run_frontier_solver(
                    solver_gr, graph, n_nodes
                )

                agree = (
                    "ok"
                    if abs(cut_v1 - cut_ck) < 1e-9 and abs(cut_v1 - cut_gr) < 1e-9
                    else "MISMATCH"
                )

                print(
                    f"{display:<24} {sup:>8} {n_nodes:>4} {n_edges:>5} "
                    f"{ms_v1:>8.1f} {ms_ck:>8.1f} {ms_gr:>8.1f} "
                    f"{st_v1:>8} {st_ck:>8} {st_gr:>8} "
                    f"{order_v1:<22} {agree:>6}"
                )

                rows.append(_FrontierDPRow(
                    tiling_key=tiling_key,
                    tiling_display=display,
                    supercell=sup,
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    cut_v1=cut_v1,
                    cut_ck=cut_ck,
                    cut_gr=cut_gr,
                    agree=agree,
                    runtime_ms_v1=round(ms_v1, 3),
                    runtime_ms_ck=round(ms_ck, 3),
                    runtime_ms_gr=round(ms_gr, 3),
                    n_states_v1=st_v1,
                    n_states_ck=st_ck,
                    n_states_gr=st_gr,
                    order_v1=order_v1,
                ))

    print(_FDP_SEP)

    mismatches = [r for r in rows if r.agree != "ok"]
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} cut-value mismatch(es) detected!")
        for r in mismatches:
            print(
                f"  {r.tiling_key} {r.supercell} n={r.n_nodes}: "
                f"v1={r.cut_v1} ck={r.cut_ck} gr={r.cut_gr}"
            )
    else:
        print("\nAll cut values agree across all three solvers.")

    if rows:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_render_frontier_dp_markdown(rows), encoding="utf-8")
        print(f"Markdown written to {args.md_out}")


# ===========================================================================
# frontier-order subcommand
# ===========================================================================

_FO_DEFAULT_SIZES: list[int] = [12, 24, 36, 48, 60, 72, 84, 96]
_FO_HEURISTICS: tuple[str, ...] = ("natural", "ck", "greedy", "geometric")

_FO_TILING_DISPLAY: dict[str, str] = {
    "hexagonal": "6³ (hexagonal)",
    "square": "4⁴ (square)",
    "truncated_square": "4.8² (truncated square)",
    "truncated_trihexagonal": "4.6.12 (truncated trihexagonal)",
    "triangular": "3⁶ (triangular)",
    "kagome": "3.6.3.6 (kagome)",
    "truncated_hexagonal": "3.12.12 (truncated hexagonal)",
    "elongated_triangular": "3³.4² (elongated triangular)",
    "rhombitrihexagonal": "3.4.6.4 (rhombitrihexagonal)",
    "snub_square": "3².4.3.4 (snub square)",
    "snub_hexagonal": "3⁴.6 (snub hexagonal)",
}

# Full catalogue with verbose display names for frontier-order
_FO_TILINGS: list[tuple[str, str]] = [
    (k, _FO_TILING_DISPLAY.get(k, k)) for k, _ in _ALL_TILINGS
]

_FoEntry = tuple[int, str, dict[str, int]]
_FoSection = tuple[str, str, int, int, list[_FoEntry]]


def _fo_winner(widths: dict[str, int]) -> str:
    """Return slash-joined names of heuristics tied for minimum frontier width.

    Args:
        widths: Mapping from heuristic name to its maximum frontier width.

    Returns:
        Slash-joined string of winning heuristic names.
    """
    min_w = min(widths.values())
    return "/".join(h for h in _FO_HEURISTICS if widths[h] == min_w)


def _render_frontier_order_markdown(sections: list[_FoSection]) -> str:
    """Render the frontier-order comparison markdown document.

    Args:
        sections: Per-tiling data collected by ``_cmd_frontier_order``.

    Returns:
        Full markdown document as a string.
    """
    lines: list[str] = [
        "# Frontier Order Comparison",
        "",
        "Compares vertex-ordering heuristics by **maximum frontier width k**"
        " across all Archimedean tilings and supercells.",
        "DP memory scales as O(2^k), so the heuristic with the smallest k"
        " is preferred to avoid out-of-memory errors.",
        "",
        "Heuristics: `natural` (ascending node ID), `ck` (Cuthill-McKee),"
        " `greedy` (min-degree elimination), `geometric` (best geometric sweep).",
        "",
        "## Per-tiling detail",
        "",
    ]

    for _, tiling_display, n_v, coord, entries in sections:
        if not entries:
            continue
        lines += [
            f"### {tiling_display}",
            "",
            f"n_v = {n_v}, coordination = {coord}",
            "",
            "| n | supercell | natural | ck | greedy | geometric | winner"
            " | memory (best k) |",
            "|---|-----------|---------|----|---------|-----------|----|---------|",
        ]
        for n, supercell, widths in entries:
            w = _fo_winner(widths)
            best_k = min(widths.values())
            mem_gb = estimate_peak_memory_bytes(best_k, n) / 1e9
            lines.append(
                f"| {n} | {supercell} |"
                f" {widths['natural']} | {widths['ck']} |"
                f" {widths['greedy']} | {widths['geometric']} | {w} | {mem_gb:.1f} GB |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _cmd_frontier_order(args: argparse.Namespace) -> None:
    """Run the frontier-order comparison subcommand.

    Args:
        args: Parsed CLI namespace with target_sizes, max_n, tilings, md_out.
    """
    target_sizes = _resolve_target_sizes(args, _FO_DEFAULT_SIZES)
    selected = _selected_tilings(args, _FO_TILINGS)
    sections: list[_FoSection] = []

    for tiling_key, tiling_display in selected:
        tiling = registry.get(tiling_key)
        n_v = tiling.n_vertices
        coord = tiling.coordination
        print(f"\n[{tiling_display}] n_v={n_v}", flush=True)
        entries: list[_FoEntry] = []

        for target_n in target_sizes:
            matrices = list(base.valid_supercell_matrices(tiling, target_n))
            for matrix in matrices:
                graph = tiling.graph(matrix)
                n = graph.number_of_nodes()
                supercell = _format_supercell_upper(matrix)
                widths = {h: compute_frontier_width(graph, h) for h in _FO_HEURISTICS}
                entries.append((n, supercell, widths))
                print(
                    f"  n={n:>3} {supercell:<14}"
                    + "  ".join(f"{h}={widths[h]}" for h in _FO_HEURISTICS),
                    flush=True,
                )

        sections.append((tiling_key, tiling_display, n_v, coord, entries))

    md = _render_frontier_order_markdown(sections)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(md, encoding="utf-8")
    print(f"\nMarkdown written to {args.md_out}", flush=True)


# ===========================================================================
# numba-jit subcommand — @njit kernel stays here (benchmark artifact)
# ===========================================================================

try:
    from numba import njit as _njit  # type: ignore[import-untyped]
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    from archimono.assignment.solvers import _bruteforce, _scoring
    from archimono.assignment.solvers._heuristic_base import (
        HeuristicAssignmentSolver,
    )
    from archimono.assignment.solvers._jit import (
        bruteforce_maxcut_jit as _bruteforce_maxcut_jit,
    )
    from archimono.assignment.solvers._jit import (
        maxcut_value_jit as _maxcut_value_jit,
    )

    @_njit(cache=True)  # type: ignore[untyped-decorator]
    def _swap_delta_jit(
        i: int,
        j: int,
        labels: np.ndarray,
        adj_neighbors: np.ndarray,
        adj_weights: np.ndarray,
        adj_offsets: np.ndarray,
    ) -> float:
        """JIT-compiled swap-delta kernel for the heuristic inner loop.

        Args:
            i: Index of the first node to swap.
            j: Index of the second node to swap.
            labels: Per-node binary labels array (int64[n]).
            adj_neighbors: CSR neighbor indices (int64[total_entries]).
            adj_weights: CSR edge weights (float64[total_entries]).
            adj_offsets: CSR row offsets (int64[n+1]).

        Returns:
            Change in cut value if nodes i and j were swapped.
        """
        li = labels[i]
        lj = labels[j]
        if li == lj:
            return 0.0

        delta = 0.0
        for idx in range(adj_offsets[i], adj_offsets[i + 1]):
            nb = adj_neighbors[idx]
            wt = adj_weights[idx]
            if nb == j:
                continue
            if labels[nb] == li:
                delta += wt
            else:
                delta -= wt
        for idx in range(adj_offsets[j], adj_offsets[j + 1]):
            nb = adj_neighbors[idx]
            wt = adj_weights[idx]
            if nb == i:
                continue
            if labels[nb] == lj:
                delta += wt
            else:
                delta -= wt
        return delta


def _edges_to_arrays(
    edges: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an edge list to (src, dst, weights) numpy arrays.

    Args:
        edges: List of (u, v, weight) edge tuples.

    Returns:
        Tuple of (src int64 array, dst int64 array, weights float64 array).
    """
    if not edges:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64)
    arr = np.array(edges)
    src = arr[:, 0].astype(np.int64)
    dst = arr[:, 1].astype(np.int64)
    wt = arr[:, 2].astype(np.float64)
    return src, dst, wt


def _adjacency_to_csr(
    adj: dict[int, list[tuple[int, float]]],
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an adjacency dict to CSR arrays.

    Args:
        adj: Adjacency dict mapping each node to its (neighbor, weight) list.
        n_nodes: Total number of nodes.

    Returns:
        Tuple of (neighbors int64 array, weights float64 array,
        offsets int64[n+1] array).
    """
    offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    neighbors_list: list[int] = []
    weights_list: list[float] = []
    for node in range(n_nodes):
        entries = adj.get(node, [])
        offsets[node + 1] = offsets[node] + len(entries)
        for nb, wt in entries:
            neighbors_list.append(nb)
            weights_list.append(wt)
    return (
        np.array(neighbors_list, dtype=np.int64),
        np.array(weights_list, dtype=np.float64),
        offsets,
    )


def _bench_maxcut_value(name: str, supercell: tuple[int, int], n_iters: int) -> None:
    """Benchmark the maxcut_value kernel (Python vs JIT).

    Args:
        name: Tiling registry key.
        supercell: Supercell tuple (a, b) or (a,).
        n_iters: Number of timed repetitions.
    """
    tiling = registry.get(name)
    graph = tiling.graph(supercell)
    n = graph.number_of_nodes()
    edges = _scoring.weighted_edge_list(graph)
    labels = [0] * (n // 2) + [1] * (n - n // 2)

    src, dst, wt = _edges_to_arrays(edges)
    labels_np = np.array(labels, dtype=np.int64)

    _maxcut_value_jit(labels_np, src, dst, wt)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _scoring.maxcut_value(labels, edges)
    py_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _maxcut_value_jit(labels_np, src, dst, wt)
    jit_time = time.perf_counter() - t0

    speedup = py_time / jit_time if jit_time > 0 else float("inf")
    print(
        f"  maxcut_value  {name:25s} {supercell}  n={n:3d}  "
        f"py={py_time:.4f}s  jit={jit_time:.4f}s  {speedup:.1f}x"
    )


def _bench_swap_delta(name: str, supercell: tuple[int, int], n_iters: int) -> None:
    """Benchmark the swap_delta kernel (Python vs JIT).

    Args:
        name: Tiling registry key.
        supercell: Supercell tuple.
        n_iters: Number of timed repetitions.
    """
    tiling = registry.get(name)
    graph = tiling.graph(supercell)
    n = graph.number_of_nodes()
    edges = _scoring.weighted_edge_list(graph)
    adjacency = HeuristicAssignmentSolver._adjacency_weights(edges)
    labels = [0] * (n // 2) + [1] * (n - n // 2)
    labels_np = np.array(labels, dtype=np.int64)
    adj_nb, adj_wt, adj_off = _adjacency_to_csr(adjacency, n)

    i, j = 0, n - 1

    _swap_delta_jit(i, j, labels_np, adj_nb, adj_wt, adj_off)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        HeuristicAssignmentSolver._swap_delta_cut_value(
            i=i, j=j, labels=labels, adjacency=adjacency,
        )
    py_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _swap_delta_jit(i, j, labels_np, adj_nb, adj_wt, adj_off)
    jit_time = time.perf_counter() - t0

    speedup = py_time / jit_time if jit_time > 0 else float("inf")
    print(
        f"  swap_delta    {name:25s} {supercell}  n={n:3d}  "
        f"py={py_time:.4f}s  jit={jit_time:.4f}s  {speedup:.1f}x"
    )


def _bench_bruteforce(name: str, supercell: tuple[int, int]) -> None:
    """Benchmark brute-force enumeration (Python vs JIT).

    Args:
        name: Tiling registry key.
        supercell: Supercell tuple.
    """
    tiling = registry.get(name)
    graph = tiling.graph(supercell)
    n = graph.number_of_nodes()
    n_b = n // 2
    edges = _scoring.weighted_edge_list(graph)
    src, dst, wt = _edges_to_arrays(edges)
    total = math.comb(n, n_b)

    _bruteforce_maxcut_jit(n, n_b, src, dst, wt)

    t0 = time.perf_counter()
    result_py = _bruteforce.solve_fixed_composition(
        edges=edges,
        graph=graph,
        target_composition=(n - n_b, n_b),
        progress_callback=lambda _done, _total: None,
        progress_interval=total + 1,
    )
    py_time = time.perf_counter() - t0
    py_cut = result_py[1]

    t0 = time.perf_counter()
    jit_cut = _bruteforce_maxcut_jit(n, n_b, src, dst, wt)
    jit_time = time.perf_counter() - t0

    match = "OK" if abs(py_cut - jit_cut) < 1e-9 else "MISMATCH"
    speedup = py_time / jit_time if jit_time > 0 else float("inf")
    print(
        f"  bruteforce    {name:25s} {supercell}  n={n:3d}  "
        f"C({n},{n_b})={total:>10,}  "
        f"py={py_time:.4f}s  jit={jit_time:.4f}s  "
        f"{speedup:.1f}x  [{match}]"
    )


def _cmd_numba_jit(_args: argparse.Namespace) -> None:
    """Run the numba-jit benchmark subcommand.

    Args:
        _args: Parsed CLI namespace (no subcommand-specific flags).
    """
    if not _NUMBA_AVAILABLE:
        raise SystemExit(
            "numba is required for this benchmark. "
            "Install with: pip install 'archimono[accel]'"
        )

    print("=" * 80)
    print("Numba JIT benchmark — hot paths in regression tests")
    print("=" * 80)

    print("\nWarming up JIT compilation...")
    dummy_l = np.zeros(2, dtype=np.int64)
    dummy_e = np.zeros(1, dtype=np.int64)
    dummy_w = np.ones(1, dtype=np.float64)
    dummy_o = np.array([0, 0], dtype=np.int64)
    _maxcut_value_jit(dummy_l, dummy_e, dummy_e, dummy_w)
    _swap_delta_jit(0, 1, dummy_l, dummy_e, dummy_w, dummy_o)
    _bruteforce_maxcut_jit(2, 1, dummy_e, dummy_e, dummy_w)
    print("Done.\n")

    print("1. maxcut_value (100,000 calls each)")
    print("-" * 70)
    for name, sc in [
        ("triangular", (2, 2)),
        ("kagome", (2, 2)),
        ("truncated_hexagonal", (2, 2)),
    ]:
        _bench_maxcut_value(name, sc, 100_000)

    print("\n2. swap_delta (100,000 calls each)")
    print("-" * 70)
    for name, sc in [
        ("triangular", (2, 2)),
        ("kagome", (2, 2)),
        ("truncated_hexagonal", (2, 2)),
    ]:
        _bench_swap_delta(name, sc, 100_000)

    print("\n3. brute-force enumeration (single run each)")
    print("-" * 70)
    for name, sc in [
        ("triangular", (2, 2)),
        ("kagome", (2, 2)),
        ("snub_square", (2, 2)),
        ("truncated_hexagonal", (2, 2)),
    ]:
        _bench_bruteforce(name, sc)

    print("\n" + "=" * 80)
    print("Done.")


# ===========================================================================
# sample-runs subcommand
# ===========================================================================

_SR_DEFAULT_SIZES: list[int] = list(range(12, 101, 12))


def _n_runs_for_solver(args: argparse.Namespace, solver_name: str) -> int:
    """Return the run count for *solver_name* given *args*.

    In sweep mode ``args`` carries ``greedy_n_runs`` / ``annealing_n_runs``
    set by :func:`_sr_run_sweep`; otherwise falls back to ``args.n_runs``.

    Args:
        args: Parsed CLI namespace.
        solver_name: ``"GreedySolver"`` or ``"AnnealingSolver"``.

    Returns:
        Number of runs to perform for this solver.
    """
    if solver_name == "GreedySolver" and hasattr(args, "greedy_n_runs"):
        return int(args.greedy_n_runs)
    if solver_name == "AnnealingSolver" and hasattr(args, "annealing_n_runs"):
        return int(args.annealing_n_runs)
    return int(args.n_runs)


def _make_greedy_solver(args: argparse.Namespace, seed: int) -> GreedySolver:
    """Construct a GreedySolver for one run.

    Args:
        args: Parsed CLI namespace (unused hyperparameters).
        seed: RNG seed.

    Returns:
        Configured :class:`GreedySolver` instance.
    """
    return GreedySolver(n_restarts=1, seed=seed)


def _make_annealing_solver(args: argparse.Namespace, seed: int) -> AnnealingSolver:
    """Construct an AnnealingSolver with hyperparameters from *args*.

    Args:
        args: Parsed CLI namespace with annealing hyperparameter flags.
        seed: RNG seed.

    Returns:
        Configured :class:`AnnealingSolver` instance.
    """
    return AnnealingSolver(
        temperature=args.annealing_temperature,
        min_temperature=args.annealing_min_temperature,
        cooling_rate=args.annealing_cooling_rate,
        steps_per_temperature=args.annealing_steps_per_temperature,
        n_restarts=1,
        seed=seed,
    )


def _sr_run_once(
    *,
    args: argparse.Namespace,
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
        args: Parsed CLI arguments (supplies annealing hyperparameters).
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
        ``"error"`` on any exception.
    """
    if solver_name == "GreedySolver":
        solver: GreedySolver | AnnealingSolver = _make_greedy_solver(args, seed)
    else:
        solver = _make_annealing_solver(args, seed)

    t0 = time.perf_counter()
    cut_value: float | None = None
    cut_efficiency: float | None = None
    n_frustrated: int | None = None
    labels: tuple[int, ...] | None = None
    metadata: dict[str, Any] = {}
    status = "ok"
    note = ""

    try:
        result = solver.solve(graph, list(SPECIES))
        cut_value = float(result.cut_value) if result.cut_value is not None else None
        n_frustrated = result.n_frustrated
        labels = result.labels
        metadata = dict(result.metadata)
        if solver_name == "AnnealingSolver":
            metadata["temperature"] = args.annealing_temperature
            metadata["min_temperature"] = args.annealing_min_temperature
            metadata["cooling_rate"] = args.annealing_cooling_rate
        if cut_value is not None and n_edges > 0:
            cut_efficiency = cut_value / n_edges
    except Exception as exc:  # noqa: BLE001
        status = "error"
        note = str(exc)

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
    args: argparse.Namespace,
    tiling: Tiling,
    tiling_key: str,
    target_n: int,
    solver_names: list[str],
    sweep_label: str | None,
    db_conn: sqlite3.Connection,
) -> int:
    """Sample all runs for one (tiling, target_n) point and insert into the DB.

    Args:
        args: Parsed CLI arguments.
        tiling: Tiling object.
        tiling_key: Registry key.
        target_n: Target supercell size.
        solver_names: List of solver class names to run.
        sweep_label: Sweep-config label if running under ``--sweep``,
            else ``None``.
        db_conn: Open SQLite connection.

    Returns:
        Total number of runs written to the database.
    """
    matrices = valid_supercell_matrices(tiling, target_n)
    if not matrices:
        print(f"  n={target_n:>3}  skip: no valid supercell", flush=True)
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
            for run_index in range(_n_runs_for_solver(args, solver_name)):
                seed = args.seed + run_index
                run = _sr_run_once(
                    args=args,
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
                    replace=args.replace_existing,
                    graph_id_cache=graph_id_cache,
                    solver_config_id_cache=solver_config_id_cache,
                ):
                    runs_written += 1

        total_runs_this_point = sum(
            _n_runs_for_solver(args, s) for s in solver_names
        )
        print(
            f"  n={n_vertices:>3}  supercell={supercell:>10}  "
            f"solvers=[{solver_summary}]  runs={total_runs_this_point}",
            flush=True,
        )
    return runs_written


def _sr_solver_names(args: argparse.Namespace) -> list[str]:
    """Map CLI solver choices to class names.

    Args:
        args: Parsed CLI namespace with a ``solvers`` list.

    Returns:
        List of solver class name strings.
    """
    names: list[str] = []
    if "greedy" in args.solvers:
        names.append("GreedySolver")
    if "annealing" in args.solvers:
        names.append("AnnealingSolver")
    return names


def _sr_run_one(
    args: argparse.Namespace,
    *,
    sweep_label: str | None,
    db_conn: sqlite3.Connection,
) -> None:
    """Run one sampling configuration and insert runs into the DB.

    Args:
        args: Parsed CLI arguments.
        sweep_label: Optional sweep-config label for this batch.
        db_conn: Open SQLite connection.
    """
    target_sizes = _resolve_target_sizes(args, _SR_DEFAULT_SIZES)
    selected = _selected_tilings(args, _ALL_TILINGS)
    solver_names = _sr_solver_names(args)

    total_runs = 0
    for tiling_key, display in selected:
        tiling = registry.get(tiling_key)
        print(
            f"\n[{display}]  n_v={tiling.n_vertices}  coord={tiling.coordination}  "
            f"bipartite={tiling.is_bipartite}",
            flush=True,
        )
        for target_n in target_sizes:
            total_runs += _sr_sample_point(
                args=args,
                tiling=tiling,
                tiling_key=tiling_key,
                target_n=target_n,
                solver_names=solver_names,
                sweep_label=sweep_label,
                db_conn=db_conn,
            )

    db_conn.commit()
    print(f"\nDone. {total_runs} runs committed to DB.", flush=True)


def _sr_run_sweep(args: argparse.Namespace) -> None:
    """Run all SWEEP_CONFIGS in sequence, writing results to the database.

    Args:
        args: Parsed CLI arguments (supplies ``db`` path and tiling filters).
    """
    print(f"Starting sweep of {len(SWEEP_CONFIGS)} configurations", flush=True)
    t_sweep_start = time.perf_counter()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    db_conn = open_db(args.db)
    print(f"DB opened at {args.db}", flush=True)

    for cfg in SWEEP_CONFIGS:
        sweep_args = copy.copy(args)
        sweep_args.greedy_n_runs = cfg.greedy_n_runs
        sweep_args.annealing_n_runs = cfg.annealing_n_runs
        sweep_args.annealing_steps_per_temperature = (
            cfg.annealing_steps_per_temperature
        )
        print(
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
        _sr_run_one(sweep_args, sweep_label=cfg.label, db_conn=db_conn)
        elapsed = time.perf_counter() - t_cfg_start
        print(f"Config {cfg.label!r} done in {elapsed / 60:.1f} min.", flush=True)

    db_conn.close()

    total_elapsed = time.perf_counter() - t_sweep_start
    print(
        f"\nSweep complete: {len(SWEEP_CONFIGS)} configs in "
        f"{total_elapsed / 60:.1f} min.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# sample-runs: report rendering helpers (ported from sample_solver_runs.py)
# ---------------------------------------------------------------------------


def _sr_filter_runs(
    runs: list[SolveRun],
    *,
    tiling: str | None,
    solver: str | None,
    n: int | None,
    supercell: str | None = None,
) -> list[SolveRun]:
    """Return runs matching all non-None filter criteria.

    Args:
        runs: Full run list to filter.
        tiling: If set, keep only runs with this ``tiling_key`` (or ``"frustrated"``
            to keep all frustrated tilings).
        solver: If set, keep only runs with this ``solver``.
        n: If set, keep only runs with this ``n_vertices``.
        supercell: If set, keep only runs with this ``supercell``.

    Returns:
        Filtered list of :class:`~archimono.benchmarking.SolveRun` records.
    """
    result = runs
    if tiling is not None:
        if tiling.lower() == "frustrated":
            result = [r for r in result if r.tiling_key in _FRUSTRATED_TILING_KEYS]
        else:
            result = [r for r in result if r.tiling_key == tiling]
    if solver is not None:
        result = [r for r in result if r.solver == solver]
    if n is not None:
        result = [r for r in result if r.n_vertices == n]
    if supercell is not None:
        result = [r for r in result if r.supercell == supercell]
    return result


def _sr_detect_sweep_labels(runs: list[SolveRun]) -> list[str]:
    """Return sweep-config labels present in runs, in canonical SWEEP_CONFIGS order.

    Args:
        runs: Run records to inspect.

    Returns:
        Ordered list of sweep labels found in the runs; empty if none have a
        ``sweep_label`` set.
    """
    present = {r.sweep_label for r in runs if r.sweep_label is not None}
    return [cfg.label for cfg in SWEEP_CONFIGS if cfg.label in present]


def _sr_sweep_config_map() -> dict[str, SweepConfig]:
    """Return a label-keyed dict of all SWEEP_CONFIGS.

    Returns:
        Mapping from sweep-config label to :class:`~archimono.benchmarking.SweepConfig`.
    """
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
    """Build the sweep-config description block for the report header.

    Args:
        sweep_labels: Ordered sweep-config labels detected in the runs.

    Returns:
        List of markdown lines describing each detected config.
    """
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

    Args:
        tiling_runs: All runs for this tiling across all sweep configs.
        sweep_labels: Ordered sweep-config labels defining column order.
        heading: Full markdown heading line.

    Returns:
        List of markdown lines.
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
    """Build a plain aggregate table for one tiling (no sweep).

    Args:
        tiling_runs: Runs for this tiling.
        heading: Full markdown heading line.

    Returns:
        List of markdown lines.
    """
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
) -> list[str]:
    """Build markdown lines for all tiling sections.

    Args:
        runs: Run records spanning all tilings.
        sweep_labels: If set, use the compact sweep-comparison table layout.

    Returns:
        List of markdown lines.
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
        print(
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

    Returns:
        Set of ``(tiling_key, n_vertices, supercell)`` triples that qualify.
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
) -> str:
    """Build a markdown report string from a list of runs.

    Args:
        runs: Filtered run records to render.
        db_path: Database path, embedded in the report header.
        sweep_labels: If set, ordered sweep-config labels detected in the runs.

    Returns:
        Markdown string (no trailing newline).
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
    lines.extend(_sr_render_tiling_sections(runs, sweep_labels=sweep_labels))
    return "\n".join(lines)


def _sr_print_report(
    db_path: pathlib.Path,
    out_path: pathlib.Path,
    *,
    tiling: str | None,
    solver: str | None,
    n: int | None,
) -> None:
    """Load runs from the database and write markdown reports.

    Writes a full report to ``out_path`` and a greedy-hard filtered report
    alongside it. Auto-detects sweep configs and groups by config when found.

    Args:
        db_path: Path to the SQLite database.
        out_path: Destination path for the full markdown report.
        tiling: Optional tiling-key filter.
        solver: Optional solver-name filter.
        n: Optional n_vertices filter.
    """
    print(f"Loading runs from {db_path} …", flush=True)
    db = SolveRunDB(db_path)
    all_runs = db.all()
    db.close()
    print(f"Loaded {len(all_runs)} runs.", flush=True)
    if not all_runs:
        print("No runs found.")
        return

    runs = _sr_filter_runs(all_runs, tiling=tiling, solver=solver, n=n)
    if not runs:
        print("No runs matched the specified filters.")
        return
    print(f"Filtered to {len(runs)} runs.", flush=True)

    sweep_labels = _sr_detect_sweep_labels(runs) or None
    if sweep_labels:
        print(f"Detected sweep configs: {sweep_labels}", flush=True)

    print("Rendering full report …", flush=True)
    md = _sr_render_report_markdown(runs, db_path=db_path, sweep_labels=sweep_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md + "\n", encoding="utf-8")
    print(f"Report written to {out_path}", flush=True)

    if sweep_labels:
        print("Computing greedy-hard supercells …", flush=True)
        hard_keys = _sr_greedy_hard_keys(runs, sweep_labels)
        hard_runs = [
            r
            for r in runs
            if (r.tiling_key, r.n_vertices, r.supercell or "") in hard_keys
        ]
        print(
            f"Found {len(hard_keys)} greedy-hard supercells"
            f" ({len(hard_runs)} runs). Rendering …",
            flush=True,
        )
        hard_md = _sr_render_report_markdown(
            hard_runs, db_path=db_path, sweep_labels=sweep_labels,
        )
        hard_path = out_path.with_stem(out_path.stem + "-greedy-hard")
        hard_path.write_text(hard_md + "\n", encoding="utf-8")
        print(f"Greedy-hard report written to {hard_path}", flush=True)


def _cmd_sample_runs(args: argparse.Namespace) -> None:
    """Run the sample-runs subcommand.

    Args:
        args: Parsed CLI namespace with db, tilings, target_sizes, solvers, etc.
    """
    if args.report:
        _sr_print_report(
            args.db,
            args.report_out,
            tiling=args.report_tiling,
            solver=args.report_solver,
            n=args.report_n,
        )
        return

    registry._ensure_loaded()

    if args.sweep:
        _sr_run_sweep(args)
    else:
        args.db.parent.mkdir(parents=True, exist_ok=True)
        db_conn = open_db(args.db)
        print(f"DB opened at {args.db}", flush=True)
        _sr_run_one(args, sweep_label=None, db_conn=db_conn)
        db_conn.close()


# ===========================================================================
# Argument parser
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Benchmarking tools for archimono solvers and frontier DP.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # --- frontier-dp ---
    p_fdp = sub.add_parser(
        "frontier-dp",
        help="Compare FrontierExactSolver / CK / Greedy variants side-by-side.",
    )
    p_fdp.add_argument(
        "--target-sizes", type=int, nargs="*", default=None,
        help=(
            "Target sizes to benchmark. Defaults to [12, 24, 36]. "
            "Passing the flag with no values falls back to these defaults."
        ),
    )
    p_fdp.add_argument(
        "--max-n", type=int, default=40,
        help="Upper bound on target sizes when --target-sizes is omitted.",
    )
    p_fdp.add_argument(
        "--tilings", nargs="*", default=None,
        help="Subset of tiling keys. Defaults to all 11 Archimedean tilings.",
    )
    p_fdp.add_argument(
        "--md-out", type=pathlib.Path,
        default=pathlib.Path("tmp/frontier-dp-benchmark.md"),
        help="Path to write results as markdown.",
    )
    p_fdp.set_defaults(func=_cmd_frontier_dp)

    # --- frontier-order ---
    p_fo = sub.add_parser(
        "frontier-order",
        help="Compare vertex-ordering heuristics by frontier width.",
    )
    p_fo.add_argument(
        "--target-sizes", type=int, nargs="*", default=None,
        help=(
            "Target sizes to benchmark. Defaults to "
            "[12, 24, 36, 48, 60, 72, 84, 96]. Passing the flag with no values "
            "falls back to these defaults."
        ),
    )
    p_fo.add_argument(
        "--max-n", type=int, default=100,
        help="Upper bound on target sizes when --target-sizes is omitted.",
    )
    p_fo.add_argument(
        "--tilings", nargs="*", default=None,
        help="Subset of tiling keys. Defaults to all 11 Archimedean tilings.",
    )
    p_fo.add_argument(
        "--md-out", type=pathlib.Path,
        default=pathlib.Path("docs/draft/frontier-order-comparison.md"),
        help="Path to write the markdown report.",
    )
    p_fo.set_defaults(func=_cmd_frontier_order)

    # --- numba-jit ---
    p_jit = sub.add_parser(
        "numba-jit",
        help="Benchmark @njit hot-path kernels vs pure Python.",
    )
    p_jit.set_defaults(func=_cmd_numba_jit)

    # --- sample-runs ---
    p_sr = sub.add_parser(
        "sample-runs",
        help="Sample per-restart solver runs and write to a SQLite database.",
    )
    p_sr.add_argument(
        "--db", type=pathlib.Path, default=None, metavar="PATH", required=True,
        help="SQLite database path (required).",
    )
    p_sr.add_argument(
        "--target-sizes", type=int, nargs="*", default=None,
        help=(
            "Explicit target sizes. Defaults to the 12-step corpus "
            f"{_SR_DEFAULT_SIZES} truncated to --max-n."
        ),
    )
    p_sr.add_argument(
        "--max-n", type=int, default=100,
        help="Upper bound on target sizes when --target-sizes is omitted.",
    )
    p_sr.add_argument(
        "--tilings", nargs="*", default=None,
        help="Subset of tiling keys. Defaults to all 11 Archimedean tilings.",
    )
    p_sr.add_argument(
        "--solvers", nargs="*", choices=["greedy", "annealing"],
        default=["greedy", "annealing"],
        help="Which solvers to sample. Defaults to both.",
    )
    p_sr.add_argument(
        "--n-runs", type=int, default=50,
        help="Independent single-restart samples per (tiling, n, solver) point.",
    )
    p_sr.add_argument(
        "--seed", type=int, default=0,
        help="Base seed; run i uses seed + i.",
    )
    p_sr.add_argument(
        "--annealing-temperature", type=float, default=5.0,
        help="AnnealingSolver initial temperature.",
    )
    p_sr.add_argument(
        "--annealing-min-temperature", type=float, default=1e-4,
        help="AnnealingSolver min_temperature.",
    )
    p_sr.add_argument(
        "--annealing-cooling-rate", type=float, default=0.995,
        help="AnnealingSolver cooling_rate.",
    )
    p_sr.add_argument(
        "--annealing-steps-per-temperature", type=int, default=200,
        help="AnnealingSolver steps_per_temperature.",
    )
    p_sr.add_argument(
        "--sweep", action="store_true",
        help="Run the built-in fast/default/thorough sampling presets.",
    )
    p_sr.add_argument(
        "--replace-existing", action="store_true",
        help="Replace existing DB rows instead of skipping them.",
    )
    p_sr.add_argument(
        "--report", action="store_true", default=False,
        help=(
            "Load runs from --db, write markdown reports, and exit. "
            "No solving is performed."
        ),
    )
    p_sr.add_argument(
        "--report-out", type=pathlib.Path,
        default=pathlib.Path("docs/draft/solver-characterization.md"),
        metavar="PATH",
        help="Destination for the markdown file written by --report.",
    )
    p_sr.add_argument(
        "--report-tiling", default=None, metavar="KEY",
        help="Used with --report: restrict output to this tiling key.",
    )
    p_sr.add_argument(
        "--report-solver", default=None, metavar="NAME",
        help="Used with --report: restrict output to this solver name.",
    )
    p_sr.add_argument(
        "--report-n", type=int, default=None, metavar="N",
        help="Used with --report: restrict output to this n_vertices.",
    )
    p_sr.set_defaults(func=_cmd_sample_runs)

    return parser


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
