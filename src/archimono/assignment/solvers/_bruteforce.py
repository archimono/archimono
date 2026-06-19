"""Shared fixed-composition brute-force helpers for exact assignment solvers."""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import combinations

import networkx as nx

from archimono.assignment.solvers import _jit, _scoring, base

_TOLERANCE: float = 1e-9


def _solve_python(
    edges: list[base.AssignmentEdge],
    n_nodes: int,
    n_ones: int,
    total_states: int,
    progress_callback: Callable[[int, int], None] | None,
    progress_interval: int | None,
) -> tuple[float, base.AssignmentLabels | None, list[base.AssignmentLabels], int]:
    """Pure-Python brute-force enumeration with progress reporting."""
    best_cut = -math.inf
    best_count = 0
    best_labels: base.AssignmentLabels | None = None
    optimal_labels: list[base.AssignmentLabels] = []
    states_evaluated = 0

    for one_nodes in combinations(range(n_nodes), n_ones):
        states_evaluated += 1
        labels = labels_from_one_nodes(n_nodes=n_nodes, one_nodes=one_nodes)
        cut_value = _scoring.maxcut_value(labels, edges)
        labels_tuple = tuple(labels)
        if cut_value > best_cut + _TOLERANCE:
            best_cut = cut_value
            best_count = 1
            best_labels = labels_tuple
            optimal_labels = [labels_tuple]
            continue
        if math.isclose(cut_value, best_cut, rel_tol=0.0, abs_tol=_TOLERANCE):
            best_count += 1
            optimal_labels.append(labels_tuple)
            if best_labels is None or labels_tuple < best_labels:
                best_labels = labels_tuple
        if (
            progress_callback is not None
            and progress_interval is not None
            and states_evaluated % progress_interval == 0
        ):
            progress_callback(states_evaluated, total_states)

    if (
        progress_callback is not None
        and states_evaluated % (progress_interval or 1) != 0
    ):
        progress_callback(states_evaluated, total_states)

    return best_cut, best_labels, optimal_labels, best_count


def _solve_jit_two_pass(
    edges: list[base.AssignmentEdge],
    n_nodes: int,
    n_ones: int,
    total_states: int,  # noqa: ARG001
) -> tuple[float, base.AssignmentLabels | None, list[base.AssignmentLabels], int]:
    """Two-pass JIT-accelerated enumeration.

    Pass 1 (JIT): find the optimal cut value over all C(n, n_ones) states.
    Pass 2 (Python + JIT scoring): collect labels for states at the optimum.
    """
    import numpy as np

    src, dst, wt = _jit.edges_to_arrays(edges)
    best_cut = float(_jit.bruteforce_maxcut_jit(n_nodes, n_ones, src, dst, wt))

    best_labels: base.AssignmentLabels | None = None
    optimal_labels: list[base.AssignmentLabels] = []
    best_count = 0
    labels_buf = np.zeros(n_nodes, dtype=np.int64)

    for one_nodes in combinations(range(n_nodes), n_ones):
        labels_buf[:] = 0
        for node in one_nodes:
            labels_buf[node] = 1
        cut_value = float(_jit.maxcut_value_jit(labels_buf, src, dst, wt))
        if math.isclose(cut_value, best_cut, rel_tol=0.0, abs_tol=_TOLERANCE):
            labels_tuple = tuple(int(x) for x in labels_buf)
            best_count += 1
            optimal_labels.append(labels_tuple)
            if best_labels is None or labels_tuple < best_labels:
                best_labels = labels_tuple

    return best_cut, best_labels, optimal_labels, best_count


def solve_fixed_composition(
    *,
    edges: list[base.AssignmentEdge],
    graph: nx.Graph[int],
    target_composition: base.TargetComposition,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval: int | None = None,
) -> tuple[base.AssignmentLabels, float, dict[str, object]]:
    """Solve fixed-composition MAX-CUT by exhaustive enumeration.

    Args:
        edges: Weighted graph edges as ``(i, j, weight)`` triples.
        graph: Assignment graph to score.
        target_composition: Exact ``(n_0, n_1)`` label counts required in each
            candidate assignment.
        progress_callback: Optional callback receiving
            ``(states_evaluated, total_states)`` during enumeration.
        progress_interval: Optional reporting interval in number of evaluated
            states. Ignored when ``progress_callback`` is ``None``.

    Returns:
        A tuple containing one optimal assignment label tuple, the
        corresponding cut value, and metadata describing the brute-force
        search. The metadata retains every optimal assignment encountered.

    Raises:
        ValueError: If ``target_composition`` is inconsistent with ``graph``.
        RuntimeError: If no candidate assignments are evaluated.
    """
    n_nodes = graph.number_of_nodes()
    n_zeros, n_ones = target_composition
    if n_zeros + n_ones != n_nodes:
        raise ValueError(
            "target_composition must sum to the graph node count for exact "
            "fixed-composition search."
        )
    total_states = math.comb(n_nodes, n_ones)

    if _jit.HAS_NUMBA and progress_callback is None:
        best_cut, best_labels, optimal_labels, best_count = (
            _solve_jit_two_pass(edges, n_nodes, n_ones, total_states)
        )
    else:
        best_cut, best_labels, optimal_labels, best_count = (
            _solve_python(
                edges, n_nodes, n_ones, total_states,
                progress_callback, progress_interval,
            )
        )

    if best_labels is None:
        raise RuntimeError("Brute-force exact search did not evaluate any assignments.")

    return best_labels, best_cut, {
        "algorithm": "brute_force_fixed_composition",
        "n_states_evaluated": total_states,
        "n_optimal_assignments": best_count,
        "optimal_labels": optimal_labels,
    }


def labels_from_one_nodes(
    *,
    n_nodes: int,
    one_nodes: tuple[int, ...],
) -> list[int]:
    """Build dense binary labels from the selected ``1``-label nodes.

    Args:
        n_nodes: Total number of graph nodes.
        one_nodes: Node indices that should receive label ``1``.

    Returns:
        Dense ``0/1`` labels indexed by node id.
    """
    labels = [0] * n_nodes
    for node in one_nodes:
        labels[node] = 1
    return labels
