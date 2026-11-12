"""Parallel brute-force solver for fixed-composition MAX-CUT assignments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TextIO

import networkx as nx

from archimono.assignment.solvers import _bruteforce, _combinatorics, _jit, base


def solve_fixed_composition_parallel(
    *,
    edges: Sequence[base.AssignmentEdge],
    graph: nx.Graph[int],
    target_composition: base.TargetComposition,
    stream: TextIO | None = None,
) -> tuple[base.AssignmentLabels, float, dict[str, Any]]:
    """Solve fixed-composition MAX-CUT by JIT-parallelized exhaustive search.

    Scans all C(n, n_b) assignments in parallel, finding the optimum cut and
    collecting the tying ranks in the same scan, then unranks only those optima
    into labels. The scan repeats with a larger per-chunk capacity if the tying
    ranks overflow it. Without numba, falls back to the serial
    :func:`~archimono.assignment.solvers._bruteforce.solve_fixed_composition`.

    Args:
        edges: Weighted graph edges as ``(i, j, weight)`` triples.
        graph: Assignment graph to score.
        target_composition: Exact ``(n_0, n_1)`` label counts required in each
            candidate assignment.
        stream: Optional output stream to print informational warnings when
            parallel JIT is unavailable.

    Returns:
        A tuple containing one optimal assignment label tuple (lexicographically
        smallest), the corresponding cut value, and solver metadata.

    Raises:
        ValueError: If ``target_composition`` does not match graph size.
        RuntimeError: If no assignments are evaluated.
        OverflowError: If numba is available and ``graph.number_of_nodes()``
            exceeds the signed ``int64`` rank-space limit (``n >= 67``); see
            :func:`archimono.assignment.solvers._combinatorics.binomial_table`.
    """
    n_nodes = graph.number_of_nodes()
    n_zeros, n_ones = target_composition
    if n_zeros + n_ones != n_nodes:
        raise ValueError(
            "target_composition must sum to the graph node count for exact "
            "fixed-composition search."
        )
    total_states = math.comb(n_nodes, n_ones)

    if _jit.HAS_NUMBA:
        src, dst, wt = _jit.edges_to_arrays(list(edges))
        binom = _combinatorics.binomial_table(n_nodes)

        # Single parallel scan: find the optimal cut and collect every tying
        # rank in one pass, then unrank only the optima into labels.
        best_cut, tie_ranks = _jit.collect_parallel_best_and_ties(
            0, total_states, n_nodes, n_ones, binom, src, dst, wt
        )
        optimal_labels: list[base.AssignmentLabels] = []
        for rank in tie_ranks:
            one_nodes = _combinatorics.unrank_combination(
                int(rank), n_nodes, n_ones
            )
            optimal_labels.append(
                tuple(
                    _bruteforce.labels_from_one_nodes(
                        n_nodes=n_nodes, one_nodes=one_nodes
                    )
                )
            )
    else:
        if stream is not None:
            stream.write("numba is absent; falling back to serial bruteforce\n")
            stream.flush()

        labels, best_cut, meta = _bruteforce.solve_fixed_composition(
            edges=list(edges),
            graph=graph,
            target_composition=target_composition,
        )
        meta_dict = dict(meta)
        return labels, best_cut, meta_dict

    if not optimal_labels:
        raise RuntimeError("Brute-force exact search did not evaluate any assignments.")

    best_labels = min(optimal_labels)

    metadata: dict[str, Any] = {
        "algorithm": "brute_force_fixed_composition",
        "n_states_evaluated": total_states,
        "n_optimal_assignments": len(optimal_labels),
        "optimal_labels": optimal_labels,
    }

    return best_labels, best_cut, metadata
