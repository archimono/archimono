"""Optional Numba JIT kernels for performance-critical solver loops.

When numba is installed the compiled kernels accelerate the brute-force
enumeration in :mod:`_bruteforce`.  When numba is absent everything
falls back transparently to Python.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

try:
    from numba import njit  # type: ignore[import-untyped]

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


# ---------------------------------------------------------------------------
# Array conversion helpers
# ---------------------------------------------------------------------------

def edges_to_arrays(
    edges: list[tuple[int, int, float]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Convert a Python edge list to three contiguous numpy arrays.

    Args:
        edges: Weighted graph edges as ``(i, j, weight)`` triples.

    Returns:
        A tuple of ``(src, dst, weights)`` arrays.
    """
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
    """JIT-compiled weighted cut value for a binary assignment.

    Args:
        labels: Dense ``0/1`` node labels.
        edge_src: Source node indices.
        edge_dst: Destination node indices.
        edge_wt: Edge weights.

    Returns:
        The weighted cut value.
    """
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

    Args:
        n_nodes: Total number of graph nodes.
        n_ones: Number of nodes to place on one side.
        edge_src: Source node indices.
        edge_dst: Destination node indices.
        edge_wt: Edge weights.

    Returns:
        The maximum cut value over all C(n_nodes, n_ones) bipartitions.
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
