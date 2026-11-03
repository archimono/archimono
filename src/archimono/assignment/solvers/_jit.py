"""Optional Numba JIT kernels for performance-critical solver loops.

When numba is installed the compiled kernels accelerate the brute-force
enumeration in :mod:`_bruteforce`.  When numba is absent everything
falls back transparently to Python.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

try:
    from numba import get_num_threads, njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    from typing import Any

    def njit(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Identity decorator when numba is absent."""
        def _identity(fn: Any) -> Any:  # noqa: ANN401
            return fn
        if args and callable(args[0]):
            return args[0]
        return _identity

    def get_num_threads() -> int:
        """Single thread when numba is absent (parallel kernels are unused)."""
        return 1

    # Without numba ``prange`` degrades to a plain serial ``range`` and the
    # typed list to the builtin; the parallel kernels below are never called in
    # this mode (callers gate on ``HAS_NUMBA``).
    prange = range


# Number of parallel work-units per CPU thread for the chunked kernels. A small
# multiple of the thread count (rather than ``1``) keeps every thread busy while
# shrinking the ragged final chunk and the cost of a straggler thread, since
# numba's ``prange`` splits the iteration space statically with no work
# stealing.
_CHUNKS_PER_THREAD = 4

# Tolerance for treating two weighted cut values as tied, matching the serial
# brute-force solver (:data:`archimono.assignment.solvers._bruteforce._TOLERANCE`).
_TIE_TOLERANCE = 1e-9

# Initial per-chunk capacity for the tie buffers. Comfortably above the
# degeneracy (number of optimal assignments) of realistic certification cases;
# the driver transparently grows it and rescans if a chunk ever overflows.
_INITIAL_TIE_CAP = 4096


# ---------------------------------------------------------------------------
# Array conversion helpers
# ---------------------------------------------------------------------------

def edges_to_arrays(
    edges: list[tuple[int, int, float]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Convert a Python edge list to contiguous ``(src, dst, weights)`` numpy arrays."""
    if not edges:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    arr = np.array(edges)
    return (
        arr[:, 0].astype(np.int64),
        arr[:, 1].astype(np.int64),
        arr[:, 2].astype(np.float64),
    )


# ---------------------------------------------------------------------------
# JIT kernels
# ---------------------------------------------------------------------------

@njit(cache=True)  # type: ignore[untyped-decorator]
def maxcut_value_jit(
    labels: npt.NDArray[np.int64],
    edge_src: npt.NDArray[np.int64],
    edge_dst: npt.NDArray[np.int64],
    edge_wt: npt.NDArray[np.float64],
) -> float:
    """JIT-compiled weighted cut value for a binary assignment."""
    cut = 0.0
    for k in range(edge_src.shape[0]):
        if labels[edge_src[k]] != labels[edge_dst[k]]:
            cut += edge_wt[k]
    return cut


@njit(cache=True)  # type: ignore[untyped-decorator]
def bruteforce_maxcut_jit(
    n_nodes: int,
    n_ones: int,
    edge_src: npt.NDArray[np.int64],
    edge_dst: npt.NDArray[np.int64],
    edge_wt: npt.NDArray[np.float64],
) -> float:
    """Enumerate all C(n, n_ones) bipartitions, return best cut.

    Uses an iterative combination generator suitable for nopython mode.
    """
    best = -np.inf
    labels = np.zeros(n_nodes, dtype=np.int64)
    indices = np.empty(n_ones, dtype=np.int64)
    for k in range(n_ones):
        indices[k] = k

    while True:
        for k in range(n_nodes):
            labels[k] = 0
        for k in range(n_ones):
            labels[indices[k]] = 1

        cut = 0.0
        for e in range(edge_src.shape[0]):
            if labels[edge_src[e]] != labels[edge_dst[e]]:
                cut += edge_wt[e]
        if cut > best:
            best = cut

        found = False
        for k in range(n_ones - 1, -1, -1):
            if indices[k] != k + n_nodes - n_ones:
                indices[k] += 1
                for m in range(k + 1, n_ones):
                    indices[m] = indices[m - 1] + 1
                found = True
                break
        if not found:
            break

    return best


@njit(cache=True)  # type: ignore[untyped-decorator]
def unrank_into_labels(
    rank: int,
    n_nodes: int,
    n_ones: int,
    binom: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
) -> None:
    """Write the ``rank``-th fixed-composition assignment into ``labels``.

    Resets ``labels`` to all zeros, then sets the ``n_ones`` entries selected by
    the ``rank``-th ``n_ones``-subset of ``{0, ..., n_nodes - 1}`` (in
    lexicographic order) to ``1``. It is written once here and reused by every
    parallel kernel so the unrank loop is not duplicated. See
    :mod:`archimono.assignment.solvers._combinatorics` for the reference.

    Args:
        rank: Zero-based lexicographic rank in ``[0, C(n_nodes, n_ones))``.
        n_nodes: Total number of graph nodes.
        n_ones: Number of ``1``-labelled nodes (the fixed composition).
        binom: Precomputed binomial table from
            :func:`archimono.assignment.solvers._combinatorics.binomial_table`.
        labels: Length-``n_nodes`` buffer overwritten in place with dense
            ``0/1`` labels.
    """
    for t in range(n_nodes):
        labels[t] = 0
    remaining = rank
    x = 0
    for i in range(n_ones):
        cc = x
        # Count combinations whose next chosen element is exactly ``cc``; skip
        # whole blocks until ``remaining`` falls within the current block.
        while True:
            block = binom[n_nodes - cc - 1, n_ones - i - 1]
            if remaining < block:
                break
            remaining -= block
            cc += 1
        labels[cc] = 1
        x = cc + 1


@njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
def bruteforce_parallel_best_and_ties_jit(
    lo: int,
    hi: int,
    n_nodes: int,
    n_ones: int,
    n_chunks: int,
    cap: int,
    binom: npt.NDArray[np.int64],
    edge_src: npt.NDArray[np.int64],
    edge_dst: npt.NDArray[np.int64],
    edge_wt: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]:
    """Single-pass parallel best-cut search that also collects per-chunk ties.

    Scans the lexicographic rank interval ``[lo, hi)`` of the ``C(n, n_ones)``
    fixed-composition assignments exactly once, splitting it into ``n_chunks``
    contiguous chunks scored in parallel with ``prange``. Each candidate is
    unranked directly from its rank (so no predecessors are enumerated). Per
    chunk this records the local best cut and, instead of discarding the
    per-state scores, the ranks whose cut lies within
    :data:`_TIE_TOLERANCE` of that chunk's evolving best:

    * a cut more than ``_TIE_TOLERANCE`` above the local best resets the chunk's
      list (every earlier rank is now strictly worse than the global optimum can
      be);
    * a cut within the tolerance band of the local best is appended.

    The collected ranks are therefore a superset of the true global ties: any
    rank within tolerance of the *global* best is also within tolerance of its
    chunk's best at visit time, so it is never dropped. The caller
    (:func:`collect_parallel_best_and_ties`) reduces the per-chunk bests to the
    global best and filters these candidates against it. This replaces the old
    three-pass scheme (one best-finding pass plus a count/collect pass pair),
    paying the dominant per-state edge scan only once.

    Output is written into preallocated 2-D buffers (one row per chunk) rather
    than per-chunk dynamic lists: under ``prange`` each chunk touches only its
    own row, so the writes are race-free (unlike a shared ``typed.List``, whose
    concurrent ``setitem`` corrupts results). A chunk whose final tie count
    exceeds ``cap`` cannot store all its ranks; it still reports the true count,
    and the caller detects the overflow and retries with a larger ``cap``.

    Args:
        lo: Inclusive start rank of the range (assumed ``< hi``).
        hi: Exclusive end rank of the range.
        n_nodes: Total number of graph nodes.
        n_ones: Number of ``1``-labelled nodes (the fixed composition).
        n_chunks: Number of parallel chunks; clamped by the caller to
            ``1 <= n_chunks <= hi - lo``.
        cap: Per-chunk capacity of the output rows.
        binom: Precomputed binomial table from
            :func:`archimono.assignment.solvers._combinatorics.binomial_table`.
        edge_src: Source node indices.
        edge_dst: Destination node indices.
        edge_wt: Edge weights.

    Returns:
        A tuple ``(partial, counts, out_ranks, out_cuts)`` where ``partial[c]``
        is the best cut in chunk ``c`` (``-inf`` for an empty chunk),
        ``counts[c]`` is the true number of tying ranks found in chunk ``c`` (may
        exceed ``cap``), and ``out_ranks[c, :min(counts[c], cap)]`` /
        ``out_cuts[c, :min(counts[c], cap)]`` hold those ranks and their cut
        values in ascending rank order.
    """
    chunk = (hi - lo + n_chunks - 1) // n_chunks
    partial = np.full(n_chunks, -np.inf)
    counts = np.zeros(n_chunks, dtype=np.int64)
    out_ranks = np.empty((n_chunks, cap), dtype=np.int64)
    out_cuts = np.empty((n_chunks, cap), dtype=np.float64)

    for c in prange(n_chunks):
        c_lo = lo + c * chunk
        c_hi = c_lo + chunk
        if c_hi > hi:
            c_hi = hi
        cnt = 0
        local_best = -np.inf
        labels = np.zeros(n_nodes, dtype=np.int64)
        for r in range(c_lo, c_hi):
            unrank_into_labels(r, n_nodes, n_ones, binom, labels)
            cut = maxcut_value_jit(labels, edge_src, edge_dst, edge_wt)
            if cut > local_best + _TIE_TOLERANCE:
                # Strictly better: discard the prior band and start fresh.
                local_best = cut
                out_ranks[c, 0] = r
                out_cuts[c, 0] = cut
                cnt = 1
            elif cut >= local_best - _TIE_TOLERANCE:
                # Within the tie band: append if there is room (else just count;
                # the caller retries with a larger cap on overflow).
                if cnt < cap:
                    out_ranks[c, cnt] = r
                    out_cuts[c, cnt] = cut
                cnt += 1
                if cut > local_best:
                    local_best = cut
        partial[c] = local_best
        counts[c] = cnt

    return partial, counts, out_ranks, out_cuts


def collect_parallel_best_and_ties(
    lo: int,
    hi: int,
    n_nodes: int,
    n_ones: int,
    binom: npt.NDArray[np.int64],
    edge_src: npt.NDArray[np.int64],
    edge_dst: npt.NDArray[np.int64],
    edge_wt: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.int64]]:
    """Find the best cut over ``[lo, hi)`` and every tying rank, in one scan.

    Drives :func:`bruteforce_parallel_best_and_ties_jit`: it reduces the
    per-chunk bests to the global best, then filters each chunk's candidate
    ranks against that global best (skipping chunks whose local best falls below
    it). The whole ``C(n, n_ones)`` rank range is scored once per attempt; if any
    chunk overflowed its capacity the scan is retried with a larger ``cap``
    (rare: ``cap`` starts well above the degeneracy of any realistic case).

    Args:
        lo: Inclusive start rank of the range.
        hi: Exclusive end rank of the range.
        n_nodes: Total number of graph nodes.
        n_ones: Number of ``1``-labelled nodes (the fixed composition).
        binom: Precomputed binomial table from
            :func:`archimono.assignment.solvers._combinatorics.binomial_table`.
        edge_src: Source node indices.
        edge_dst: Destination node indices.
        edge_wt: Edge weights.

    Returns:
        A tuple ``(best_cut, tie_ranks)`` where ``best_cut`` is the maximum
        weighted cut (``-inf`` if ``[lo, hi)`` is empty) and ``tie_ranks`` is the
        ascending array of every rank whose cut is within :data:`_TIE_TOLERANCE`
        of ``best_cut``.
    """
    n_states = hi - lo
    if n_states <= 0:
        return float("-inf"), np.empty(0, dtype=np.int64)

    n_chunks = _CHUNKS_PER_THREAD * get_num_threads()
    if n_chunks > n_states:
        n_chunks = n_states

    cap = _INITIAL_TIE_CAP
    while True:
        partial, counts, out_ranks, out_cuts = (
            bruteforce_parallel_best_and_ties_jit(
                lo, hi, n_nodes, n_ones, n_chunks, cap,
                binom, edge_src, edge_dst, edge_wt,
            )
        )
        max_count = int(counts.max()) if counts.shape[0] > 0 else 0
        if max_count <= cap:
            break
        # A chunk held more ties than its row could store; grow and rescan.
        cap = max_count

    best = -np.inf
    for c in range(n_chunks):
        if partial[c] > best:
            best = partial[c]

    tie_ranks: list[int] = []
    for c in range(n_chunks):
        if partial[c] < best - _TIE_TOLERANCE:
            continue
        for i in range(int(counts[c])):
            if abs(out_cuts[c, i] - best) <= _TIE_TOLERANCE:
                tie_ranks.append(int(out_ranks[c, i]))

    tie_ranks.sort()
    return float(best), np.array(tie_ranks, dtype=np.int64)
