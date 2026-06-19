"""Exact fixed-composition MAX-CUT via frontier dynamic programming.

Four solver variants, differing only in vertex-ordering strategy:

- :class:`FrontierExactSolverNatural`: processes vertices in natural
  (sorted) order — no candidate evaluation overhead.
- :class:`FrontierExactSolver`: evaluates geometric sweep candidates and
  picks the one with the smallest maximum frontier size.
- :class:`FrontierExactSolverCK`: uses Cuthill-McKee ordering only.
- :class:`FrontierExactSolverGreedy`: uses greedy minimum-degree
  elimination only.

All four share an identical frontier-DP back-end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf
from typing import NamedTuple

import networkx as nx
from networkx.utils.rcm import cuthill_mckee_ordering as _cuthill_mckee_ordering

from archimono.assignment.solvers import _validation, base

# frontier_bitmask packed as int, n_ones_used
StateKey = tuple[int, int]

# Switch to two-pass reconstruction above this node count to avoid
# accumulating O(n * 2^k) parent-map objects in RAM.
_TWO_PASS_THRESHOLD = 36


class _FrontierStats(NamedTuple):
    max_frontier_size: int
    dp_cost: float


@dataclass(frozen=True, slots=True)
class _OrderCandidate:
    """One candidate vertex ordering for the frontier dynamic program."""

    name: str
    order: tuple[int, ...]
    max_frontier_size: int
    dp_cost: float


class _FrontierDPBase(base.AssignmentSolver):
    """Shared solve() logic for all frontier-DP solver variants.

    Subclasses implement :meth:`_select_order` to choose a vertex ordering.
    """

    def _target_composition(
        self,
        n_nodes: int,
        n_b: int | None,
    ) -> base.TargetComposition:
        """Return the exact fixed-composition target for the frontier DP.

        Args:
            n_nodes: Total number of vertices in the graph.
            n_b: Desired count of label-1 vertices. Defaults to ``n_nodes // 2``.

        Returns:
            A ``(n_a, n_b)`` pair summing to ``n_nodes``.

        Raises:
            ValueError: If ``n_b`` is not a valid integer in ``[0, n_nodes]``.
        """
        if n_b is None:
            n_b = n_nodes // 2
        if not isinstance(n_b, int) or isinstance(n_b, bool):
            raise ValueError("n_b must be an integer count of 1 labels.")
        if n_b < 0 or n_b > n_nodes:
            raise ValueError(f"n_b must satisfy 0 <= n_b <= {n_nodes}.")
        return (n_nodes - n_b, n_b)

    def _select_order(self, graph: nx.Graph[int]) -> _OrderCandidate:
        """Return the chosen ordering candidate.

        Args:
            graph: The input graph.

        Returns:
            The selected :class:`_OrderCandidate`.
        """
        raise NotImplementedError

    def solve(
        self,
        graph: nx.Graph[int],
        species: Sequence[str],
        *,
        n_b: int | None = None,
        enumerate_all: bool = False,
    ) -> base.AssignmentResult:
        """Solve a fixed-composition binary MAX-CUT instance exactly.

        Args:
            graph: The interaction graph.
            species: A two-element sequence of species labels (e.g. ``["A", "B"]``).
            n_b: Number of vertices to assign label ``species[1]``.  Defaults to
                ``graph.number_of_nodes() // 2``.
            enumerate_all: If ``True``, collect every optimal assignment and
                store them as ``"all_labels"`` in the result metadata, each a
                tuple of ``0/1`` ints (same encoding as ``labels``). If
                ``False`` (default), only one optimal assignment is returned
                and ``"all_labels"`` is absent from metadata.

        Returns:
            An :class:`~archimono.assignment.solvers.base.AssignmentResult` with the
            optimal assignment, cut value, and solver metadata.
        """
        _validation.validate_binary_species_problem(graph, species)

        target_composition = self._target_composition(graph.number_of_nodes(), n_b)
        candidate = self._select_order(graph)
        all_labels_int, cut_value, dp_metadata = _solve_frontier_dp(
            graph=graph,
            order=candidate.order,
            n_b=target_composition[1],
            enumerate_all=enumerate_all,
        )
        labels = all_labels_int[0]
        n_frustrated = sum(
            1 for u, v in graph.edges() if labels[int(u)] == labels[int(v)]
        )
        species_tuple = tuple(species)

        extra: dict[str, object] = {}
        if enumerate_all:
            # 0/1 int tuples (same encoding as ``labels``) so consumers such as
            # certification.write_case_json can pack them directly; the
            # species mapping is recorded separately in ``label_mapping``.
            extra["all_labels"] = [tuple(assignment) for assignment in all_labels_int]

        return base.AssignmentResult(
            labels=labels,
            objective_value=cut_value,
            cut_value=cut_value,
            n_frustrated=n_frustrated,
            solver=self.__class__.__name__,
            metadata={
                "species": species_tuple,
                "label_mapping": {0: species[0], 1: species[1]},
                "target_composition": target_composition,
                "order_name": candidate.name,
                "max_frontier_size": candidate.max_frontier_size,
                "dp_cost": candidate.dp_cost,
                **extra,
                **dp_metadata,
            },
        )


class FrontierExactSolverNatural(_FrontierDPBase):
    """Frontier DP solver using plain natural (sorted-node-id) ordering.

    Processes vertices in ascending node-ID order with no candidate
    evaluation overhead. Useful as a baseline or when the graph's natural
    numbering already reflects a good sweep direction.
    """

    def _select_order(self, graph: nx.Graph[int]) -> _OrderCandidate:
        """Return the natural ascending node ordering.

        Args:
            graph: The input graph.

        Returns:
            An :class:`_OrderCandidate` for the natural ordering.
        """
        order = tuple(sorted(int(n) for n in graph.nodes()))
        stats = _frontier_stats(graph, order)
        return _OrderCandidate(
            name="natural",
            order=order,
            max_frontier_size=stats.max_frontier_size,
            dp_cost=stats.dp_cost,
        )


class FrontierExactSolver(_FrontierDPBase):
    """Frontier DP solver using geometric sweep candidate selection.

    Evaluates the natural order, its reverse, and up to four geometric
    sweeps derived from node ``pos`` attributes when available. Picks the
    candidate with the smallest maximum frontier size.
    """

    def _select_order(self, graph: nx.Graph[int]) -> _OrderCandidate:
        """Return the best geometric-sweep candidate ordered by max frontier size.

        Args:
            graph: The input graph.

        Returns:
            The :class:`_OrderCandidate` with the smallest max frontier size.
        """
        return _best_geometric_order(graph)


class FrontierExactSolverCK(_FrontierDPBase):
    """Frontier DP solver that uses only Cuthill-McKee ordering.

    Skips candidate evaluation to eliminate overhead. Useful when graph
    structure is known to suit bandwidth-minimising orders.
    """

    def _select_order(self, graph: nx.Graph[int]) -> _OrderCandidate:
        """Return the Cuthill-McKee ordering candidate.

        Args:
            graph: The input graph.

        Returns:
            An :class:`_OrderCandidate` for the Cuthill-McKee ordering.
        """
        order = tuple(int(n) for n in _cuthill_mckee_ordering(graph))
        stats = _frontier_stats(graph, order)
        return _OrderCandidate(
            name="cuthill_mckee",
            order=order,
            max_frontier_size=stats.max_frontier_size,
            dp_cost=stats.dp_cost,
        )


class FrontierExactSolverGreedy(_FrontierDPBase):
    """Frontier DP solver that uses only greedy min-degree ordering.

    Skips candidate evaluation to eliminate overhead. Tends to outperform
    geometric sweeps on dense or irregular graphs.
    """

    def _select_order(self, graph: nx.Graph[int]) -> _OrderCandidate:
        """Return the greedy minimum-degree ordering candidate.

        Args:
            graph: The input graph.

        Returns:
            An :class:`_OrderCandidate` for the greedy min-degree ordering.
        """
        order = _greedy_min_degree_order(graph)
        stats = _frontier_stats(graph, order)
        return _OrderCandidate(
            name="greedy_min_degree",
            order=order,
            max_frontier_size=stats.max_frontier_size,
            dp_cost=stats.dp_cost,
        )


def _best_geometric_order(graph: nx.Graph[int]) -> _OrderCandidate:
    """Choose a vertex ordering from geometric sweep candidates.

    Evaluates the natural order, its reverse, and geometric sweeps derived
    from node ``pos`` attributes when available.  Returns the candidate with
    the smallest maximum frontier size (ties broken by name).

    Args:
        graph: The input graph.

    Returns:
        The :class:`_OrderCandidate` with the smallest max frontier size.
    """
    candidates: list[tuple[str, tuple[int, ...]]] = []
    seen_orders: set[tuple[int, ...]] = set()

    def add_candidate(name: str, order: Sequence[int]) -> None:
        order_tuple = tuple(int(node) for node in order)
        if order_tuple in seen_orders:
            return
        seen_orders.add(order_tuple)
        candidates.append((name, order_tuple))

    nodes = tuple(sorted(int(node) for node in graph.nodes()))
    add_candidate("natural", nodes)
    add_candidate("natural_reverse", tuple(reversed(nodes)))

    raw_positions = {
        int(node): graph.nodes[node].get("pos")
        for node in graph.nodes()
    }
    if all(pos is not None for pos in raw_positions.values()):
        positions = {
            node: pos
            for node, pos in raw_positions.items()
            if pos is not None
        }
        add_candidate(
            "x_then_y",
            sorted(
                positions,
                key=lambda node: (
                    float(positions[node][0]),
                    float(positions[node][1]),
                    node,
                ),
            ),
        )
        add_candidate(
            "y_then_x",
            sorted(
                positions,
                key=lambda node: (
                    float(positions[node][1]),
                    float(positions[node][0]),
                    node,
                ),
            ),
        )
        add_candidate(
            "x_then_y_reverse",
            tuple(
                reversed(
                    sorted(
                        positions,
                        key=lambda node: (
                            float(positions[node][0]),
                            float(positions[node][1]),
                            node,
                        ),
                    )
                )
            ),
        )
        add_candidate(
            "y_then_x_reverse",
            tuple(
                reversed(
                    sorted(
                        positions,
                        key=lambda node: (
                            float(positions[node][1]),
                            float(positions[node][0]),
                            node,
                        ),
                    )
                )
            ),
        )

    evaluated = [
        _OrderCandidate(
            name=name,
            order=order,
            max_frontier_size=stats.max_frontier_size,
            dp_cost=stats.dp_cost,
        )
        for name, order in candidates
        for stats in (_frontier_stats(graph, order),)
    ]
    return min(
        evaluated,
        key=lambda c: (c.max_frontier_size, c.name),
    )


def _greedy_min_degree_order(graph: nx.Graph[int]) -> tuple[int, ...]:
    """Return a greedy minimum-degree elimination ordering.

    At each step selects the unprocessed node with the fewest edges into the
    remaining unprocessed subgraph. This is a standard treewidth heuristic
    and tends to produce small frontiers on irregular sparse graphs.

    Args:
        graph: The input graph (not modified).

    Returns:
        A tuple of node IDs in elimination order.
    """
    remaining: set[int] = {int(n) for n in graph.nodes()}
    degree: dict[int, int] = {
        int(n): sum(1 for nb in graph.neighbors(n) if int(nb) in remaining)
        for n in graph.nodes()
    }
    order: list[int] = []
    while remaining:
        node = min(remaining, key=lambda n: (degree[n], n))
        order.append(node)
        remaining.remove(node)
        for nb in graph.neighbors(node):
            nb_int = int(nb)
            if nb_int in remaining:
                degree[nb_int] -= 1
    return tuple(order)


def _frontier_stats(
    graph: nx.Graph[int],
    order: Sequence[int],
) -> _FrontierStats:
    """Return frontier statistics for one ordering.

    Computes the maximum frontier size and the DP-cost estimate
    ``sum(2**frontier_size_at_step)``, which is proportional to the total
    number of DP states explored.

    Args:
        graph: The input graph.
        order: Vertex processing order.

    Returns:
        A :class:`_FrontierStats` named tuple with ``max_frontier_size``
        (int) and ``dp_cost`` (float).
    """
    position = {int(node): index for index, node in enumerate(order)}
    last_neighbor_position = {
        int(node): max(
            (position[int(neighbor)] for neighbor in graph.neighbors(node)),
            default=position[int(node)],
        )
        for node in order
    }

    max_frontier_size = 0
    dp_cost = 0.0
    for step in range(len(order)):
        active_before_size = sum(
            1
            for node in order[:step]
            if last_neighbor_position[int(node)] >= step
        )
        has_future_neighbor = any(
            position[int(neighbor)] > step
            for neighbor in graph.neighbors(order[step])
        )
        frontier_size = active_before_size + int(has_future_neighbor)
        if frontier_size > max_frontier_size:
            max_frontier_size = frontier_size
        dp_cost += 2.0 ** frontier_size
    return _FrontierStats(max_frontier_size=max_frontier_size, dp_cost=dp_cost)


def _backtrack_all(
    parent_maps: list[dict[StateKey, list[tuple[StateKey, int]]]],
    final_states: list[StateKey],
    n_nodes: int,
) -> list[tuple[int, ...]]:
    """Enumerate all optimal assignments by DFS through the parent DAG.

    Args:
        parent_maps: Per-step maps from each DP state to its list of
            ``(previous_state, label)`` parent edges.
        final_states: All final DP states that achieved the optimal cut value.
        n_nodes: Total number of vertices (length of each returned assignment).

    Returns:
        All distinct per-step label sequences that achieve the optimal cut.
        Each element is a tuple of integer labels indexed by processing order.
    """
    results: list[tuple[int, ...]] = []
    buffer: list[int] = [0] * n_nodes

    def _dfs(step: int, state: StateKey) -> None:
        if step < 0:
            results.append(tuple(buffer))
            return
        for prev_state, label in parent_maps[step][state]:
            buffer[step] = label
            _dfs(step - 1, prev_state)

    for final_state in final_states:
        _dfs(n_nodes - 1, final_state)
    return results


def _reconstruct_two_pass(
    *,
    dp_layers: list[dict[StateKey, float]],
    after_sources: list[tuple[int | None, ...]],
    earlier_neighbors: list[tuple[tuple[int, float], ...]],
    n_b: int,
    n_nodes: int,
    best_cut: float,
    enumerate_all: bool,
) -> list[tuple[int, ...]]:
    """Reconstruct optimal assignments by backward scan through score layers.

    Used by the two-pass path (``n_nodes > _TWO_PASS_THRESHOLD``) to avoid
    storing parent maps.  At each backward step the function recomputes which
    ``(prev_state, label)`` transitions produced the known next state, reading
    the delta from ``earlier_neighbors`` and the new frontier mask from
    ``after_sources``.

    Args:
        dp_layers: Score-only DP layers from the forward pass; ``dp_layers[t]``
            is the dict of states reachable after processing step ``t``.
        after_sources: Pre-computed frontier source indices per step.
        earlier_neighbors: Pre-computed earlier-neighbor ``(index, weight)``
            pairs per step.
        n_b: Target count of label-1 assignments.
        n_nodes: Total number of vertices.
        best_cut: Optimal cut value from the forward pass.
        enumerate_all: If ``True``, collect every optimal assignment; if
            ``False``, return the first one found.

    Returns:
        All (or one) optimal per-step label sequences, each a tuple of
        integers indexed by processing order.
    """
    final_layer = dp_layers[-1]
    if enumerate_all:
        optimal_final_states = [
            state for state, score in final_layer.items()
            if state[1] == n_b and score == best_cut
        ]
    else:
        optimal_final_states = [
            next(
                state for state, score in final_layer.items()
                if state[1] == n_b and score == best_cut
            )
        ]

    results: list[tuple[int, ...]] = []
    buffer: list[int] = [0] * n_nodes

    def _dfs(step: int, current_state: StateKey, target_score: float) -> None:
        if step < 0:
            results.append(tuple(buffer))
            return
        prev_layer = dp_layers[step - 1] if step > 0 else {(0, 0): 0.0}
        current_bitmask, current_ones = current_state
        for prev_state, prev_score in prev_layer.items():
            prev_bitmask, prev_ones = prev_state
            for label in (0, 1):
                if prev_ones + label != current_ones:
                    continue
                next_bitmask = 0
                for i, source in enumerate(after_sources[step]):
                    bit = label if source is None else (prev_bitmask >> source) & 1
                    next_bitmask |= bit << i
                if next_bitmask != current_bitmask:
                    continue
                delta = sum(
                    w for idx, w in earlier_neighbors[step]
                    if ((prev_bitmask >> idx) & 1) != label
                )
                if abs(prev_score + delta - target_score) < 1e-9:
                    buffer[step] = label
                    _dfs(step - 1, prev_state, prev_score)
                    if not enumerate_all and results:
                        return
            if not enumerate_all and results:
                return

    for final_state in optimal_final_states:
        _dfs(n_nodes - 1, final_state, final_layer[final_state])
        if not enumerate_all and results:
            break

    return results


_FrontierTables = tuple[
    list[tuple[int | None, ...]],   # after_sources
    list[tuple[tuple[int, float], ...]],  # earlier_neighbors
]


def _build_frontier_tables(
    graph: nx.Graph[int],
    order: Sequence[int],
) -> _FrontierTables:
    """Pre-compute per-step frontier source and neighbor tables.

    Args:
        graph: The interaction graph with optional ``weight`` edge attributes.
        order: Vertex processing order (a permutation of all node IDs).

    Returns:
        A pair ``(after_sources, earlier_neighbors)`` where
        ``after_sources[t]`` encodes how to build the frontier bitmask after
        step ``t``, and ``earlier_neighbors[t]`` lists the
        ``(frontier_index, weight)`` pairs for edges already processed.
    """
    position = {int(node): index for index, node in enumerate(order)}
    last_neighbor_position = {
        int(node): max(
            (position[int(neighbor)] for neighbor in graph.neighbors(node)),
            default=position[int(node)],
        )
        for node in order
    }
    after_sources: list[tuple[int | None, ...]] = []
    earlier_neighbors: list[tuple[tuple[int, float], ...]] = []

    for step, node in enumerate(order):
        active_before = tuple(
            index
            for index, previous in enumerate(order[:step])
            if last_neighbor_position[int(previous)] >= step
        )
        previous_index = {pos: idx for idx, pos in enumerate(active_before)}

        future_positions = tuple(
            position[int(neighbor)]
            for neighbor in graph.neighbors(node)
            if position[int(neighbor)] > step
        )
        active_after_positions = [
            pos
            for pos in active_before
            if last_neighbor_position[int(order[pos])] > step
        ]
        if future_positions:
            active_after_positions.append(step)
        after_sources.append(
            tuple(
                None if pos == step else previous_index[pos]
                for pos in active_after_positions
            )
        )
        earlier_neighbors.append(
            tuple(
                (
                    previous_index[position[int(neighbor)]],
                    float(graph.edges[int(neighbor), int(node)].get("weight", 1.0)),
                )
                for neighbor in graph.neighbors(node)
                if position[int(neighbor)] < step
            )
        )

    return after_sources, earlier_neighbors


def _forward_value_only(
    after_sources: list[tuple[int | None, ...]],
    earlier_neighbors: list[tuple[tuple[int, float], ...]],
    n_nodes: int,
    n_b: int,
) -> tuple[float, dict[str, int]]:
    """Forward DP pass returning only the optimal cut value.

    Single rolling layer — O(2^k) peak memory regardless of ``n_nodes``.

    Args:
        after_sources: Per-step frontier source indices from
            :func:`_build_frontier_tables`.
        earlier_neighbors: Per-step earlier-neighbor ``(index, weight)``
            pairs from :func:`_build_frontier_tables`.
        n_nodes: Total number of vertices.
        n_b: Target count of label-1 assignments.

    Returns:
        A pair ``(best_cut, metadata)``.

    Raises:
        RuntimeError: If the DP exhausts all states or finds no valid assignment.
    """
    dp: dict[StateKey, float] = {(0, 0): 0.0}
    n_states_explored = 1
    max_states_retained = 1

    for step in range(n_nodes):
        next_dp: dict[StateKey, float] = {}
        remaining_vertices = n_nodes - step - 1
        for state, score in dp.items():
            labels_bitmask, n_ones_used = state
            for label in (0, 1):
                next_ones = n_ones_used + label
                if next_ones > n_b:
                    continue
                if next_ones + remaining_vertices < n_b:
                    continue
                delta = sum(
                    w for idx, w in earlier_neighbors[step]
                    if ((labels_bitmask >> idx) & 1) != label
                )
                next_bitmask = 0
                for i, source in enumerate(after_sources[step]):
                    bit = label if source is None else (
                        (labels_bitmask >> source) & 1
                    )
                    next_bitmask |= bit << i
                next_state: StateKey = (next_bitmask, next_ones)
                next_score = score + delta
                if next_score > next_dp.get(next_state, -inf):
                    next_dp[next_state] = next_score
        if not next_dp:
            raise RuntimeError(
                f"{__name__}: frontier DP exhausted all states unexpectedly."
            )
        dp = next_dp
        n_states_explored += len(dp)
        max_states_retained = max(max_states_retained, len(dp))

    final = [s for s in dp if s[1] == n_b]
    if not final:
        raise RuntimeError(
            f"{__name__}: no assignment found at target composition n_b={n_b}."
        )
    return max(dp[s] for s in final), {
        "n_states_explored": n_states_explored,
        "max_states_retained": max_states_retained,
    }


def _forward_two_pass(
    after_sources: list[tuple[int | None, ...]],
    earlier_neighbors: list[tuple[tuple[int, float], ...]],
    n_nodes: int,
    n_b: int,
    enumerate_all: bool,
) -> tuple[list[tuple[int, ...]], float, dict[str, int]]:
    """Forward DP pass storing score layers, then backward reconstruction.

    Avoids parent-map memory by storing only float scores per layer and
    re-deriving transitions during backward DFS.

    Args:
        after_sources: Per-step frontier source indices.
        earlier_neighbors: Per-step earlier-neighbor ``(index, weight)`` pairs.
        n_nodes: Total number of vertices.
        n_b: Target count of label-1 assignments.
        enumerate_all: If ``True``, collect all optimal assignments.

    Returns:
        A triple ``(assignments_in_order, best_cut, metadata)``.

    Raises:
        RuntimeError: If the DP exhausts all states or finds no valid assignment.
    """
    dp_layers: list[dict[StateKey, float]] = []
    dp: dict[StateKey, float] = {(0, 0): 0.0}
    n_states_explored = 1
    max_states_retained = 1

    for step in range(n_nodes):
        next_dp: dict[StateKey, float] = {}
        remaining_vertices = n_nodes - step - 1
        for state, score in dp.items():
            labels_bitmask, n_ones_used = state
            for label in (0, 1):
                next_ones = n_ones_used + label
                if next_ones > n_b:
                    continue
                if next_ones + remaining_vertices < n_b:
                    continue
                delta = sum(
                    w for idx, w in earlier_neighbors[step]
                    if ((labels_bitmask >> idx) & 1) != label
                )
                next_bitmask = 0
                for i, source in enumerate(after_sources[step]):
                    bit = label if source is None else (
                        (labels_bitmask >> source) & 1
                    )
                    next_bitmask |= bit << i
                next_state: StateKey = (next_bitmask, next_ones)
                next_score = score + delta
                if next_score > next_dp.get(next_state, -inf):
                    next_dp[next_state] = next_score
        if not next_dp:
            raise RuntimeError(
                f"{__name__}: frontier DP exhausted all states unexpectedly."
            )
        dp_layers.append(next_dp)
        dp = next_dp
        n_states_explored += len(dp)
        max_states_retained = max(max_states_retained, len(dp))

    final_candidates = [(s, sc) for s, sc in dp.items() if s[1] == n_b]
    if not final_candidates:
        raise RuntimeError(
            f"{__name__}: no assignment found at target composition n_b={n_b}."
        )
    best_cut = max(sc for _, sc in final_candidates)
    assignments = _reconstruct_two_pass(
        dp_layers=dp_layers,
        after_sources=after_sources,
        earlier_neighbors=earlier_neighbors,
        n_b=n_b,
        n_nodes=n_nodes,
        best_cut=best_cut,
        enumerate_all=enumerate_all,
    )
    return assignments, best_cut, {
        "n_states_explored": n_states_explored,
        "max_states_retained": max_states_retained,
    }


def _forward_single_pass(
    after_sources: list[tuple[int | None, ...]],
    earlier_neighbors: list[tuple[tuple[int, float], ...]],
    n_nodes: int,
    n_b: int,
    enumerate_all: bool,
) -> tuple[list[tuple[int, ...]], float, dict[str, int]]:
    """Forward DP pass storing parent maps for direct backtracking.

    Stores one parent-pointer list per state per step. Used when ``n_nodes``
    is small enough that the O(n * 2^k) parent-map memory is acceptable.

    Args:
        after_sources: Per-step frontier source indices.
        earlier_neighbors: Per-step earlier-neighbor ``(index, weight)`` pairs.
        n_nodes: Total number of vertices.
        n_b: Target count of label-1 assignments.
        enumerate_all: If ``True``, collect all optimal assignments.

    Returns:
        A triple ``(assignments_in_order, best_cut, metadata)``.

    Raises:
        RuntimeError: If the DP exhausts all states or finds no valid assignment.
    """
    dp: dict[StateKey, float] = {(0, 0): 0.0}
    parent_maps: list[dict[StateKey, list[tuple[StateKey, int]]]] = []
    n_states_explored = 1
    max_states_retained = 1

    for step in range(n_nodes):
        next_dp: dict[StateKey, float] = {}
        parent_map: dict[StateKey, list[tuple[StateKey, int]]] = {}
        remaining_vertices = n_nodes - step - 1
        for state, score in dp.items():
            labels_bitmask, n_ones_used = state
            for label in (0, 1):
                next_ones = n_ones_used + label
                if next_ones > n_b:
                    continue
                if next_ones + remaining_vertices < n_b:
                    continue
                delta = sum(
                    w for idx, w in earlier_neighbors[step]
                    if ((labels_bitmask >> idx) & 1) != label
                )
                next_bitmask = 0
                for i, source in enumerate(after_sources[step]):
                    bit = label if source is None else (
                        (labels_bitmask >> source) & 1
                    )
                    next_bitmask |= bit << i
                next_state: StateKey = (next_bitmask, next_ones)
                next_score = score + delta
                current_best = next_dp.get(next_state, -inf)
                if next_score > current_best:
                    next_dp[next_state] = next_score
                    parent_map[next_state] = [(state, label)]
                elif next_score == current_best and enumerate_all:
                    parent_map[next_state].append((state, label))
        if not next_dp:
            raise RuntimeError(
                f"{__name__}: frontier DP exhausted all states unexpectedly."
            )
        parent_maps.append(parent_map)
        dp = next_dp
        n_states_explored += len(dp)
        max_states_retained = max(max_states_retained, len(dp))

    final_candidates = [(s, sc) for s, sc in dp.items() if s[1] == n_b]
    if not final_candidates:
        raise RuntimeError(
            f"{__name__}: no assignment found at target composition n_b={n_b}."
        )
    best_cut = max(sc for _, sc in final_candidates)
    if enumerate_all:
        optimal_finals = [s for s, sc in final_candidates if sc == best_cut]
    else:
        optimal_finals = [next(s for s, sc in final_candidates if sc == best_cut)]
    assignments = _backtrack_all(parent_maps, optimal_finals, n_nodes)
    return assignments, best_cut, {
        "n_states_explored": n_states_explored,
        "max_states_retained": max_states_retained,
    }


def _solve_frontier_dp(
    *,
    graph: nx.Graph[int],
    order: Sequence[int],
    n_b: int,
    enumerate_all: bool = False,
    value_only: bool = False,
) -> tuple[list[base.AssignmentLabels], float, dict[str, int]]:
    """Dispatch a fixed-composition MAX-CUT solve to the appropriate forward pass.

    Args:
        graph: The interaction graph with optional ``weight`` edge attributes.
        order: Vertex processing order (a permutation of all node IDs).
        n_b: Target count of label-1 assignments.
        enumerate_all: If ``True``, backtrack all optimal paths. Ignored when
            ``value_only=True``.
        value_only: If ``True``, return only the cut value (empty label list).
            Uses a single rolling layer — O(2^k) peak memory.

    Returns:
        A triple ``(all_labels, best_cut, metadata)``. ``all_labels`` is empty
        when ``value_only=True``.

    Raises:
        ValueError: If ``order`` is not a valid permutation of the graph nodes,
            or if both ``value_only`` and ``enumerate_all`` are ``True``.
        RuntimeError: If the DP unexpectedly exhausts all states.
    """
    if value_only and enumerate_all:
        raise ValueError("value_only and enumerate_all are mutually exclusive.")
    n_nodes = graph.number_of_nodes()
    if len(order) != n_nodes or set(int(n) for n in order) != set(
        int(n) for n in graph.nodes()
    ):
        raise ValueError(
            "Frontier DP order must be a permutation of every graph node exactly once."
        )

    after_sources, earlier_neighbors = _build_frontier_tables(graph, order)

    if value_only:
        best_cut, metadata = _forward_value_only(
            after_sources, earlier_neighbors, n_nodes, n_b
        )
        return [], best_cut, metadata

    if n_nodes > _TWO_PASS_THRESHOLD:
        assignments, best_cut, metadata = _forward_two_pass(
            after_sources, earlier_neighbors, n_nodes, n_b, enumerate_all
        )
    else:
        assignments, best_cut, metadata = _forward_single_pass(
            after_sources, earlier_neighbors, n_nodes, n_b, enumerate_all
        )

    all_labels_by_node: list[base.AssignmentLabels] = []
    for assigned_in_order in assignments:
        labels_by_node = [0] * n_nodes
        for step, node in enumerate(order):
            labels_by_node[int(node)] = assigned_in_order[step]
        all_labels_by_node.append(tuple(labels_by_node))

    return all_labels_by_node, best_cut, metadata


def compute_max_cut_value(
    graph: nx.Graph[int],
    *,
    n_b: int | None = None,
    heuristic: str = "geometric",
) -> float:
    """Return the exact fixed-composition MAX-CUT value without labels.

    Uses a single rolling DP layer — O(2^k) peak memory where k is the maximum
    frontier width, independent of the number of vertices.

    Args:
        graph: The interaction graph with optional ``weight`` edge attributes.
            Nodes must be integers (or castable to int).
        n_b: Number of vertices to assign label 1. Defaults to
            ``graph.number_of_nodes() // 2``.
        heuristic: Vertex-ordering strategy. One of ``"natural"``, ``"ck"``,
            ``"greedy"``, or ``"geometric"`` (default). See
            :func:`compute_frontier_width` for descriptions. Use
            :func:`select_heuristic` to choose the best ordering before calling.

    Returns:
        The optimal cut value k*.

    Raises:
        ValueError: If ``n_b`` is out of range or ``heuristic`` is not recognised.
    """
    n_nodes = graph.number_of_nodes()
    if n_b is None:
        n_b = n_nodes // 2
    if not isinstance(n_b, int) or isinstance(n_b, bool):
        raise ValueError("n_b must be an integer count of 1 labels.")
    if n_b < 0 or n_b > n_nodes:
        raise ValueError(f"n_b must satisfy 0 <= n_b <= {n_nodes}.")
    if heuristic not in _FRONTIER_HEURISTICS:
        raise ValueError(
            f"Unknown heuristic {heuristic!r}. "
            f"Expected one of: {sorted(_FRONTIER_HEURISTICS)!r}."
        )
    if heuristic == "geometric":
        order: tuple[int, ...] = _best_geometric_order(graph).order
    elif heuristic == "natural":
        order = tuple(sorted(int(n) for n in graph.nodes()))
    elif heuristic == "ck":
        order = tuple(int(n) for n in _cuthill_mckee_ordering(graph))
    else:  # greedy
        order = _greedy_min_degree_order(graph)
    _, best_cut, _ = _solve_frontier_dp(
        graph=graph,
        order=order,
        n_b=n_b,
        value_only=True,
    )
    return best_cut


_FRONTIER_HEURISTICS: frozenset[str] = frozenset(
    {"natural", "ck", "greedy", "geometric"}
)

_SELECT_HEURISTIC_PRIORITY: tuple[str, ...] = ("geometric", "ck", "greedy")


def compute_frontier_width(graph: nx.Graph[int], heuristic: str) -> int:
    """Return the maximum frontier width for a given vertex-ordering heuristic.

    The maximum frontier width *k* is the peak bitmask width during DP
    traversal.  Memory scales as O(2^k), so minimising *k* is key to
    avoiding out-of-memory errors.

    Args:
        graph: The input graph.  Nodes must be integers or castable to int.
        heuristic: Ordering strategy.  One of ``"natural"`` (ascending
            node-ID), ``"ck"`` (Cuthill-McKee), ``"greedy"`` (greedy
            min-degree elimination), or ``"geometric"`` (best-of-N
            geometric sweep).

    Returns:
        Maximum frontier width k.

    Raises:
        ValueError: If *heuristic* is not a recognised value.
    """
    if heuristic not in _FRONTIER_HEURISTICS:
        raise ValueError(
            f"Unknown heuristic {heuristic!r}. "
            f"Expected one of: {sorted(_FRONTIER_HEURISTICS)!r}."
        )
    if heuristic == "geometric":
        return _best_geometric_order(graph).max_frontier_size
    if heuristic == "natural":
        order: tuple[int, ...] = tuple(sorted(int(n) for n in graph.nodes()))
    elif heuristic == "ck":
        order = tuple(int(n) for n in _cuthill_mckee_ordering(graph))
    elif heuristic == "greedy":
        order = _greedy_min_degree_order(graph)
    else:
        raise AssertionError(f"unreachable: {heuristic!r}")
    return _frontier_stats(graph, order).max_frontier_size


def estimate_peak_memory_bytes(k: int, n: int) -> int:
    """Estimate peak memory for a frontier DP solve.

    Uses the formula ``2 × 2^k × (n // 2 + 1) × 144`` bytes, where 2 accounts
    for two coexisting DP layers during each step transition, ``2^k`` is the
    bitmask state count, ``(n // 2 + 1)`` is the label-count multiplicity, and
    144 bytes is the estimated cost per Python dict entry.

    Args:
        k: Maximum frontier width.
        n: Number of vertices in the graph.

    Returns:
        Estimated peak memory in bytes.
    """
    return 2 * (1 << k) * (n // 2 + 1) * 144


def select_heuristic(graph: nx.Graph[int]) -> tuple[str, int]:
    """Select the best vertex-ordering heuristic by minimum frontier width.

    Evaluates ``"geometric"``, ``"ck"``, and ``"greedy"`` heuristics. Geometric
    already subsumes natural as one of its sweep candidates. Ties are broken in
    the order geometric > ck > greedy.

    Args:
        graph: The input graph. Nodes must be integers or castable to int.

    Returns:
        A ``(heuristic_name, min_frontier_width)`` pair where
        ``heuristic_name`` is the tie-broken winner and ``min_frontier_width``
        is the corresponding maximum frontier width k.
    """
    widths = {h: compute_frontier_width(graph, h) for h in _SELECT_HEURISTIC_PRIORITY}
    min_k = min(widths.values())
    for name in _SELECT_HEURISTIC_PRIORITY:
        if widths[name] == min_k:
            return name, min_k
    raise AssertionError("unreachable")
