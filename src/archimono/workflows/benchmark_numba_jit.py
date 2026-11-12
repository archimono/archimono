"""Numba-JIT benchmark workflow: @njit kernels vs pure-Python equivalents."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from archimono.assignment.solvers._jit import edges_to_arrays as _edges_to_arrays
from archimono.tilings import registry
from archimono.workflows._benchmark_common import _print


@dataclass(slots=True)
class NumbaJitConfig:
    """Configuration for the numba-JIT benchmark workflow."""


@dataclass(slots=True)
class NumbaJitResult:
    """Result from the numba-JIT benchmark workflow.

    Attributes:
        numba_available: Whether numba was available and benchmarks ran.
    """

    numba_available: bool


try:
    from numba import njit as _njit
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

    # NOTE: benchmark-local shadow of
    # HeuristicAssignmentSolver._swap_delta_cut_value; if the production
    # swap-delta algorithm changes, this kernel must be updated to match or
    # the benchmark silently measures stale logic.
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


def _bench_maxcut_value(
    name: str,
    supercell: tuple[int, int],
    n_iters: int,
    *,
    stream: TextIO | None,
) -> None:
    """Benchmark the maxcut_value kernel (Python vs JIT)."""
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
    _print(
        stream,
        f"  maxcut_value  {name:25s} {supercell}  n={n:3d}  "
        f"py={py_time:.4f}s  jit={jit_time:.4f}s  {speedup:.1f}x"
    )


def _bench_swap_delta(
    name: str,
    supercell: tuple[int, int],
    n_iters: int,
    *,
    stream: TextIO | None,
) -> None:
    """Benchmark the swap_delta kernel (Python vs JIT)."""
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
    _print(
        stream,
        f"  swap_delta    {name:25s} {supercell}  n={n:3d}  "
        f"py={py_time:.4f}s  jit={jit_time:.4f}s  {speedup:.1f}x"
    )


def _bench_bruteforce(
    name: str,
    supercell: tuple[int, int],
    *,
    stream: TextIO | None,
) -> None:
    """Benchmark brute-force enumeration (Python vs JIT)."""
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
    _print(
        stream,
        f"  bruteforce    {name:25s} {supercell}  n={n:3d}  "
        f"C({n},{n_b})={total:>10,}  "
        f"py={py_time:.4f}s  jit={jit_time:.4f}s  "
        f"{speedup:.1f}x  [{match}]"
    )


def run_numba_jit(
    _config: NumbaJitConfig,
    *,
    stream: TextIO | None = None,
) -> NumbaJitResult:
    """Run the numba-JIT benchmark workflow.

    Raises:
        ImportError: If numba is not installed.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError(
            "numba is required for this benchmark. "
            "Install with: pip install 'archimono[accel]'"
        )

    _print(stream, "=" * 80)
    _print(stream, "Numba JIT benchmark — hot paths in regression tests")
    _print(stream, "=" * 80)

    _print(stream, "\nWarming up JIT compilation...")
    dummy_l = np.zeros(2, dtype=np.int64)
    dummy_e = np.zeros(1, dtype=np.int64)
    dummy_w = np.ones(1, dtype=np.float64)
    dummy_o = np.array([0, 0], dtype=np.int64)
    _maxcut_value_jit(dummy_l, dummy_e, dummy_e, dummy_w)
    _swap_delta_jit(0, 1, dummy_l, dummy_e, dummy_w, dummy_o)
    _bruteforce_maxcut_jit(2, 1, dummy_e, dummy_e, dummy_w)
    _print(stream, "Done.\n")

    _print(stream, "1. maxcut_value (100,000 calls each)")
    _print(stream, "-" * 70)
    for name, sc in [
        ("triangular", (3, 3)),
        ("kagome", (2, 2)),
        ("truncated_hexagonal", (2, 2)),
    ]:
        _bench_maxcut_value(name, sc, 100_000, stream=stream)

    _print(stream, "\n2. swap_delta (100,000 calls each)")
    _print(stream, "-" * 70)
    for name, sc in [
        ("triangular", (3, 3)),
        ("kagome", (2, 2)),
        ("truncated_hexagonal", (2, 2)),
    ]:
        _bench_swap_delta(name, sc, 100_000, stream=stream)

    _print(stream, "\n3. brute-force enumeration (single run each)")
    _print(stream, "-" * 70)
    for name, sc in [
        ("triangular", (3, 3)),
        ("kagome", (2, 2)),
        ("snub_square", (2, 2)),
        ("truncated_hexagonal", (2, 2)),
    ]:
        _bench_bruteforce(name, sc, stream=stream)

    _print(stream, "\n" + "=" * 80)
    _print(stream, "Done.")
    return NumbaJitResult(numba_available=True)
