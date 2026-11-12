"""Fast invariants for parallel fixed-composition brute-force search."""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.assignment.solvers._bruteforce import solve_fixed_composition
from archimono.assignment.solvers.parallel_bruteforce import (
    solve_fixed_composition_parallel,
)


def test_parallel_bruteforce_rejects_mismatched_composition() -> None:
    """Target compositions not summing to graph node count are rejected."""
    graph = nx.cycle_graph(5)
    edges = [(u, v, 1.0) for u, v in graph.edges()]
    with pytest.raises(
        ValueError, match="target_composition must sum to the graph node count"
    ):
        solve_fixed_composition_parallel(
            edges=edges,
            graph=graph,
            target_composition=(2, 2),  # sums to 4, graph has 5 nodes
        )


# (graph, label) cases with many tied optima. Sparse/structured graphs have a
# large optimum degeneracy whose tying assignments are spread across the rank
# space and across parallel chunks -- the case that a race-prone tie collector
# silently corrupts (dropping true optima, inventing false ones).
_HIGH_DEGENERACY_GRAPHS = [
    ("empty_12", nx.empty_graph(12)),  # every assignment ties; forces cap growth
    ("two_hexagons", nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))),
    ("matching_14", nx.Graph([(2 * i, 2 * i + 1) for i in range(7)])),
    ("sparse_random_14", nx.gnp_random_graph(14, 0.2, seed=7)),
]


@pytest.mark.parametrize("name,graph", _HIGH_DEGENERACY_GRAPHS)
def test_parallel_matches_serial_on_high_degeneracy(
    name: str, graph: nx.Graph
) -> None:
    """Parallel search must reproduce the serial solver's full optimum set.

    Regression guard for the tie-collection path: the parallel and serial
    solvers must agree on the optimal cut value, the chosen (lexicographically
    smallest) labels, the degeneracy count, and the exact set of optimal
    assignments -- even when the optimum is highly degenerate.
    """
    n = graph.number_of_nodes()
    n_ones = n // 2
    composition = (n - n_ones, n_ones)
    edges = [(int(u), int(v), 1.0) for u, v in graph.edges()]

    labels_s, cut_s, meta_s = solve_fixed_composition(
        edges=edges, graph=graph, target_composition=composition
    )
    labels_p, cut_p, meta_p = solve_fixed_composition_parallel(
        edges=edges, graph=graph, target_composition=composition
    )

    assert cut_p == pytest.approx(cut_s)
    assert labels_p == labels_s
    assert meta_p["n_optimal_assignments"] == meta_s["n_optimal_assignments"]
    assert set(meta_p["optimal_labels"]) == set(meta_s["optimal_labels"])
