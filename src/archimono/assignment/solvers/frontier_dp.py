"""Exact fixed-composition MAX-CUT via frontier dynamic programming.

:class:`FrontierExactSolver` evaluates geometric sweep candidates and picks
the one with the smallest maximum frontier size.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import comb, inf
from typing import NamedTuple

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.utils.rcm import cuthill_mckee_ordering as _cuthill_mckee_ordering

from archimono.assignment.solvers import _scoring, _validation, base

# frontier_bitmask packed as int, n_ones_used
StateKey = tuple[int, int]


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


class FrontierExactSolver(base.AssignmentSolver):
    """Frontier DP solver using geometric sweep candidate selection.

    Evaluates the natural order, its reverse, and up to four geometric
    sweeps derived from node ``pos`` attributes when available. Picks the
    candidate with the smallest maximum frontier size.
    """

    def _target_composition(
        self,
        n_nodes: int,
        n_b: int | None,
    ) -> base.TargetComposition:
        """Return the exact composition target, defaulting ``n_b`` to ``n_nodes // 2``.

        Raises:
            ValueError: If ``n_b`` is not a valid integer in ``[0, n_nodes]``.
        """
        return _validation.resolve_fixed_composition(n_nodes, n_b)

    def _select_order(self, graph: nx.Graph[int]) -> _OrderCandidate:
        """Return the best geometric-sweep candidate ordered by max frontier size."""
        return _best_geometric_order(graph)

    def solve(
        self,
        graph: nx.Graph[int],
        species: Sequence[str],
        *,
        n_b: int | None = None,
        enumerate_all: bool = False,
        max_optima: int | None = None,
    ) -> base.AssignmentResult:
        """Solve a fixed-composition binary MAX-CUT instance exactly.

        Args:
            graph: The interaction graph.
            species: A two-element sequence of species labels (e.g. ``["A", "B"]``).
            n_b: Number of vertices to assign label ``species[1]``.  Defaults to
                ``graph.number_of_nodes() // 2``.
            enumerate_all: If ``True``, collect every optimal assignment and
                store them as ``"all_labels"`` in the result metadata: a
                node-indexed ``uint8`` ndarray of shape ``(n_optima, n_nodes)``
                with 0/1 entries (same encoding as ``labels``). If ``False``
                (default), only one optimal assignment is returned and
                ``"all_labels"`` is absent from metadata.
            max_optima: Maximum number of optimal assignments to materialize
                when ``enumerate_all`` is ``True``.  ``None`` means no cap.
                Ignored when ``enumerate_all`` is ``False``.

        Returns:
            An :class:`~archimono.assignment.solvers.base.AssignmentResult` with the
            optimal assignment, cut value, and solver metadata.

        Raises:
            ValueError: If the number of optima exceeds ``max_optima`` during
                enumeration.
        """
        _validation.validate_binary_species_problem(graph, species)

        target_composition = self._target_composition(graph.number_of_nodes(), n_b)
        candidate = self._select_order(graph)
        all_labels_arr, cut_value, dp_metadata = _solve_frontier_dp(
            graph=graph,
            order=candidate.order,
            n_b=target_composition[1],
            enumerate_all=enumerate_all,
            max_optima=max_optima if enumerate_all else None,
        )
        labels = tuple(int(x) for x in all_labels_arr[0])
        n_frustrated = _scoring.count_frustrated_edges(graph, labels)
        species_tuple = tuple(species)

        extra: dict[str, object] = {}
        if enumerate_all:
            # Node-indexed uint8 array (rows are 0/1 assignments, same encoding
            # as ``labels``) so consumers such as certification.write_case_json
            # can pack them directly; the species mapping is recorded separately
            # in ``label_mapping``.
            extra["all_labels"] = all_labels_arr

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


def _best_geometric_order(graph: nx.Graph[int]) -> _OrderCandidate:
    """Choose a vertex ordering from geometric sweep candidates.

    Evaluates the natural order, its reverse, and geometric sweeps derived
    from node ``pos`` attributes when available.  Returns the candidate with
    the smallest maximum frontier size (ties broken by name).
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


def _frontier_sizes(
    graph: nx.Graph[int],
    order: Sequence[int],
) -> tuple[int, ...]:
    """Return the per-step frontier sizes for one ordering.

    The frontier size at step ``t`` is the number of already-processed
    vertices still relevant to edges at or beyond step ``t``, plus the
    current vertex when it has unprocessed neighbors.
    """
    position = {int(node): index for index, node in enumerate(order)}
    last_neighbor_position = {
        int(node): max(
            (position[int(neighbor)] for neighbor in graph.neighbors(node)),
            default=position[int(node)],
        )
        for node in order
    }

    sizes: list[int] = []
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
        sizes.append(active_before_size + int(has_future_neighbor))
    return tuple(sizes)


def _frontier_stats(
    graph: nx.Graph[int],
    order: Sequence[int],
) -> _FrontierStats:
    """Return frontier statistics for one ordering.

    Computes the maximum frontier size and the DP-cost estimate
    ``sum(2**frontier_size_at_step)``, which is proportional to the total
    number of DP states explored.
    """
    sizes = _frontier_sizes(graph, order)
    return _FrontierStats(
        max_frontier_size=max(sizes, default=0),
        dp_cost=sum(2.0**size for size in sizes),
    )


def _stack_optima_rows(
    rows: list[npt.NDArray[np.uint8]],
    n_nodes: int,
) -> npt.NDArray[np.uint8]:
    """Stack accumulated optimal-assignment rows into a 2-D ``uint8`` array.

    Returns:
        A ``(len(rows), n_nodes)`` ``uint8`` array, or a ``(0, n_nodes)`` array
        when ``rows`` is empty.
    """
    if not rows:
        return np.empty((0, n_nodes), dtype=np.uint8)
    return np.stack(rows)


def _enumerate_from_parent_dag(
    parent_maps: list[dict[StateKey, list[tuple[StateKey, int]]]],
    optimal_finals: list[StateKey],
    n_nodes: int,
    max_optima: int | None = None,
) -> npt.NDArray[np.uint8]:
    """Enumerate optimal assignments by DFS through the stored parent DAG.

    Backward phase (consumer): recovers the witness assignments consistent with
    the forward DP's ``best_cut`` by walking parent pointers. It never
    re-optimizes; the forward pass is the single source of truth for the cut
    value. Completed assignments are accumulated directly as ``uint8`` rows —
    no intermediate list of Python int-tuples is built.

    Args:
        parent_maps: Per-step maps from each DP state to its list of
            ``(previous_state, label)`` parent edges.
        optimal_finals: All final DP states that achieved the optimal cut value.
        n_nodes: Total number of vertices (width of each returned assignment).
        max_optima: Maximum number of assignments to materialize.  ``None``
            means no cap.

    Returns:
        A 2-D ``uint8`` array of shape ``(n_optima, n_nodes)`` whose columns are
        indexed by processing order; each row is one optimal assignment.

    Raises:
        ValueError: If the number of materialized paths exceeds ``max_optima``.
    """
    rows: list[npt.NDArray[np.uint8]] = []
    buffer: npt.NDArray[np.uint8] = np.zeros(n_nodes, dtype=np.uint8)

    # Explicit-stack DFS: recursion depth would equal n_nodes, which overflows
    # the interpreter stack on the large thin graphs the frontier DP targets.
    # stack[i] iterates the parent edges consumed at step n_nodes - 1 - i.
    for final_state in optimal_finals:
        if n_nodes == 0:
            rows.append(buffer.copy())
            if max_optima is not None and len(rows) > max_optima:
                raise ValueError(
                    f"Enumeration exceeded max_optima={max_optima}. "
                    f"Raise max_optima to materialize more optimal "
                    f"assignments."
                )
            continue
        stack: list[Iterator[tuple[StateKey, int]]] = [
            iter(parent_maps[n_nodes - 1][final_state])
        ]
        while stack:
            step = n_nodes - len(stack)
            entry = next(stack[-1], None)
            if entry is None:
                stack.pop()
                continue
            prev_state, label = entry
            buffer[step] = label
            if step == 0:
                rows.append(buffer.copy())
                if max_optima is not None and len(rows) > max_optima:
                    raise ValueError(
                        f"Enumeration exceeded max_optima={max_optima}. "
                        f"Raise max_optima to materialize more optimal "
                        f"assignments."
                    )
            else:
                stack.append(iter(parent_maps[step - 1][prev_state]))
    return _stack_optima_rows(rows, n_nodes)


_FrontierTables = tuple[
    list[tuple[int | None, ...]],   # after_sources
    list[tuple[tuple[int, float], ...]],  # earlier_neighbors
]


def _build_frontier_tables(
    graph: nx.Graph[int],
    order: Sequence[int],
) -> _FrontierTables:
    """Pre-compute per-step frontier source and neighbor tables.

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


def _compute_value_forward(
    after_sources: list[tuple[int | None, ...]],
    earlier_neighbors: list[tuple[tuple[int, float], ...]],
    n_nodes: int,
    n_b: int,
) -> tuple[float, dict[str, int]]:
    """Forward DP pass returning only the optimal cut value.

    Forward phase (source of truth): computes ``best_cut`` over all partial
    assignments represented by each frontier state. Keeps two rolling layers,
    each holding at most ``2^k × (n_b + 1)`` states.

    Returns:
        A pair ``(best_cut, metadata)``.

    Raises:
        RuntimeError: If the DP exhausts all states or finds no valid assignment.
    """
    dp: dict[StateKey, float] = {(0, 0): 0.0}
    n_states_explored = 1
    max_states_retained = 1

    # NOTE: this transition loop is intentionally forked from
    # _compute_value_with_parents (which additionally records parent edges).
    # Merging them behind a callback would slow the hottest loop in the
    # package; any pruning or delta change here must be mirrored there.
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


def _compute_value_with_parents(
    after_sources: list[tuple[int | None, ...]],
    earlier_neighbors: list[tuple[tuple[int, float], ...]],
    n_nodes: int,
    n_b: int,
    enumerate_all: bool,
) -> tuple[
    float,
    list[dict[StateKey, list[tuple[StateKey, int]]]],
    list[StateKey],
    dict[str, int],
]:
    """Forward DP pass computing ``best_cut`` and storing parent pointers.

    Forward phase (source of truth): computes ``best_cut`` over all partial
    assignments while recording, per step, the ``(prev_state, label)`` parents
    of each retained state. It does **not** backtrack; assignment recovery is a
    separate consumer. Stores one parent-pointer list per state per step; this
    is the sole reconstruction forward pass for all assignment-returning solves.

    Args:
        after_sources: Per-step frontier source indices.
        earlier_neighbors: Per-step earlier-neighbor ``(index, weight)`` pairs.
        n_nodes: Total number of vertices.
        n_b: Target count of label-1 assignments.
        enumerate_all: If ``True``, store tie parents so every optimal path can
            later be enumerated, and return every optimal final state.

    Returns:
        A quadruple ``(best_cut, parent_maps, optimal_finals, metadata)`` where
        ``optimal_finals`` lists the final DP states achieving ``best_cut`` (all
        of them when ``enumerate_all``; otherwise the first one).

    Raises:
        RuntimeError: If the DP exhausts all states or finds no valid assignment.
    """
    dp: dict[StateKey, float] = {(0, 0): 0.0}
    parent_maps: list[dict[StateKey, list[tuple[StateKey, int]]]] = []
    n_states_explored = 1
    max_states_retained = 1

    # NOTE: this transition loop is intentionally forked from
    # _compute_value_forward (the value-only path without parent bookkeeping).
    # Any pruning or delta change here must be mirrored there.
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
    return best_cut, parent_maps, optimal_finals, {
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
    max_optima: int | None = None,
) -> tuple[npt.NDArray[np.uint8], float, dict[str, int]]:
    """Dispatch a fixed-composition MAX-CUT solve to the appropriate forward pass.

    Args:
        graph: The interaction graph with optional ``weight`` edge attributes.
        order: Vertex processing order (a permutation of all node IDs).
        n_b: Target count of label-1 assignments.
        enumerate_all: If ``True``, backtrack all optimal paths. Ignored when
            ``value_only=True``.
        value_only: If ``True``, return only the cut value (no labels). Keeps
            two rolling layers, each holding at most ``2^k × (n_b + 1)`` states.
        max_optima: Maximum number of optimal assignments to materialize.
            ``None`` means no cap. Only meaningful when ``enumerate_all=True``.

    Returns:
        A triple ``(all_labels, best_cut, metadata)`` where ``all_labels`` is a
        node-indexed ``uint8`` array of shape ``(n_optima, n_nodes)`` whose rows
        are 0/1 assignments. It has zero rows when ``value_only=True``.

    Raises:
        ValueError: If ``order`` is not a valid permutation of the graph nodes,
            or if both ``value_only`` and ``enumerate_all`` are ``True``, or if
            the number of optima exceeds ``max_optima``.
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
        best_cut, metadata = _compute_value_forward(
            after_sources, earlier_neighbors, n_nodes, n_b
        )
        return np.empty((0, n_nodes), dtype=np.uint8), best_cut, metadata

    best_cut, parent_maps, optimal_finals, metadata = _compute_value_with_parents(
        after_sources, earlier_neighbors, n_nodes, n_b, enumerate_all
    )
    optima_in_order = _enumerate_from_parent_dag(
        parent_maps, optimal_finals, n_nodes, max_optima=max_optima
    )

    # Remap columns from processing order to node id in one vectorized step:
    # ``all_labels_by_node[:, order[step]] = optima_in_order[:, step]``.
    order_index = np.fromiter(
        (int(node) for node in order), dtype=np.intp, count=n_nodes
    )
    all_labels_by_node: npt.NDArray[np.uint8] = np.empty_like(optima_in_order)
    all_labels_by_node[:, order_index] = optima_in_order

    return all_labels_by_node, best_cut, metadata


def compute_max_cut_value(
    graph: nx.Graph[int],
    *,
    n_b: int | None = None,
    heuristic: str = "geometric",
) -> float:
    """Return the exact fixed-composition MAX-CUT value without labels.

    Keeps two rolling DP layers, each holding at most ``2^k × (n_b + 1)`` states,
    where k is the maximum frontier width.

    Args:
        graph: The interaction graph with optional ``weight`` edge attributes.
            Nodes must be integers (or castable to int).
        n_b: Number of vertices to assign label 1. Defaults to
            ``graph.number_of_nodes() // 2``.
        heuristic: Vertex-ordering strategy. One of ``"natural"``, ``"ck"``,
            ``"greedy"``, or ``"geometric"`` (default). See
            :func:`compute_frontier_width` for descriptions. Use
            :func:`select_heuristic` to choose the best ordering before calling.

    Raises:
        ValueError: If ``n_b`` is out of range or ``heuristic`` is not recognised.
    """
    _, n_b = _validation.resolve_fixed_composition(graph.number_of_nodes(), n_b)
    _, best_cut, _ = _solve_frontier_dp(
        graph=graph,
        order=_order_for_heuristic(graph, heuristic),
        n_b=n_b,
        value_only=True,
    )
    return best_cut


_FRONTIER_HEURISTICS: frozenset[str] = frozenset(
    {"natural", "ck", "greedy", "geometric"}
)

_SELECT_HEURISTIC_PRIORITY: tuple[str, ...] = ("geometric", "ck", "greedy")


def _order_for_heuristic(
    graph: nx.Graph[int],
    heuristic: str,
) -> tuple[int, ...]:
    """Return the vertex ordering produced by an ordering heuristic.

    Args:
        graph: The input graph.  Nodes must be integers or castable to int.
        heuristic: Ordering strategy; see :func:`compute_frontier_width`.

    Raises:
        ValueError: If *heuristic* is not a recognised value.
    """
    if heuristic not in _FRONTIER_HEURISTICS:
        raise ValueError(
            f"Unknown heuristic {heuristic!r}. "
            f"Expected one of: {sorted(_FRONTIER_HEURISTICS)!r}."
        )
    if heuristic == "geometric":
        return _best_geometric_order(graph).order
    if heuristic == "natural":
        return tuple(sorted(int(n) for n in graph.nodes()))
    if heuristic == "ck":
        return tuple(int(n) for n in _cuthill_mckee_ordering(graph))
    if heuristic == "greedy":
        return _greedy_min_degree_order(graph)
    raise AssertionError(f"unreachable: {heuristic!r}")


def compute_frontier_sizes(
    graph: nx.Graph[int],
    heuristic: str,
) -> tuple[int, ...]:
    """Return the per-step frontier-size sequence for an ordering heuristic.

    The sequence has one entry per vertex in processing order; its maximum is
    the frontier width *k* (see :func:`compute_frontier_width`) and
    ``sum(2**size)`` over it is the DP cost (see :func:`compute_dp_cost`).
    It is the input to the reachability-aware value-only memory estimator
    :func:`estimate_peak_memory_bytes`, and is a cheap graph-structural
    computation — no exponential DP runs.

    Args:
        graph: The input graph.  Nodes must be integers or castable to int.
        heuristic: Ordering strategy; see :func:`compute_frontier_width`.

    Raises:
        ValueError: If *heuristic* is not a recognised value.
    """
    return _frontier_sizes(graph, _order_for_heuristic(graph, heuristic))


def _order_stats(graph: nx.Graph[int], heuristic: str) -> _FrontierStats:
    """Return the frontier-width and DP-cost stats for an ordering heuristic.

    Shared dispatch for :func:`compute_frontier_width` and
    :func:`compute_dp_cost`, which each return one field of the same
    underlying computation.

    Args:
        graph: The input graph.  Nodes must be integers or castable to int.
        heuristic: Ordering strategy; see :func:`compute_frontier_width`.

    Raises:
        ValueError: If *heuristic* is not a recognised value.
    """
    return _frontier_stats(graph, _order_for_heuristic(graph, heuristic))


def compute_frontier_width(graph: nx.Graph[int], heuristic: str) -> int:
    """Return the maximum frontier width for a given vertex-ordering heuristic.

    The maximum frontier width *k* is the peak bitmask width during DP
    traversal.  Each DP layer holds at most ``2^k × (n_b + 1)`` states, so
    minimising *k* is key to avoiding out-of-memory errors.

    Args:
        graph: The input graph.  Nodes must be integers or castable to int.
        heuristic: Ordering strategy.  One of ``"natural"`` (ascending
            node-ID), ``"ck"`` (Cuthill-McKee), ``"greedy"`` (greedy
            min-degree elimination), or ``"geometric"`` (best-of-N
            geometric sweep).

    Raises:
        ValueError: If *heuristic* is not a recognised value.
    """
    return _order_stats(graph, heuristic).max_frontier_size


def compute_dp_cost(graph: nx.Graph[int], heuristic: str) -> float:
    """Return the DP-cost estimate for a given vertex-ordering heuristic.

    The DP cost ``Σ_steps 2^frontier_size`` is the summed state count over all
    DP layers.  It is the natural memory predictor for the reconstruction path
    (see :func:`estimate_reconstruct_memory_bytes`), which retains a parent
    map per layer, and is computed for free during vertex-order selection —
    before the exponential DP runs.

    Args:
        graph: The input graph.  Nodes must be integers or castable to int.
        heuristic: Ordering strategy; see :func:`compute_frontier_width`.

    Raises:
        ValueError: If *heuristic* is not a recognised value.
    """
    return _order_stats(graph, heuristic).dp_cost


def estimate_peak_memory_bytes(frontier_sizes: Sequence[int], n_b: int) -> int:
    """Estimate peak memory in bytes for a frontier DP value-only solve.

    A DP state after step ``t`` is a ``(frontier_bitmask, n_ones)`` pair, and the
    composition pruning bounds the valid ``n_ones`` range per bitmask popcount
    ``p``, so the layer holds at most ``sum over p of C(width_t, p) * |valid
    n_ones range|`` states.  The estimate is ``2 × max_t(reachable_t) × 160``
    bytes: 2 for the two coexisting rolling layers during a step transition, and
    160 bytes per retained entry.

    The reachable count never falls below any layer's true retained size
    (every counted constraint is a necessary condition for a state to exist),
    and never exceeds the old blanket bound, so at a fixed budget this can
    only admit cases the blanket formula over-conservatively rejected.

    Args:
        frontier_sizes: Per-step frontier sizes for the chosen vertex
            ordering (see :func:`compute_frontier_sizes`); one entry per
            vertex in processing order.
        n_b: Target count of label-1 assignments.
    """
    n = len(frontier_sizes)
    max_reachable = 1
    for step, width in enumerate(frontier_sizes):
        processed = step + 1
        remaining = n - processed
        reachable = 0
        for popcount in range(width + 1):
            lowest_ones = max(popcount, n_b - remaining)
            highest_ones = min(n_b, popcount + processed - width)
            if highest_ones >= lowest_ones:
                reachable += comb(width, popcount) * (
                    highest_ones - lowest_ones + 1
                )
        max_reachable = max(max_reachable, reachable)
    return 2 * max_reachable * 160


def estimate_reconstruct_memory_bytes(dp_cost: float, n: int) -> int:
    """Estimate peak memory in bytes for a frontier-DP reconstruction solve.

    Unlike :func:`estimate_peak_memory_bytes`, which bounds the value-only
    path's two rolling DP layers, the reconstruction path (``solve()`` /
    :class:`FrontierExactSolver`) retains a parent-pointer map for *every* DP
    layer, so its memory is a sum over layers.  The predictor is therefore
    ``dp_cost = Σ_steps 2^frontier_size``, not the single-layer peak width ``k``.

    This estimate covers DP-state dictionaries and parent-pointer maps
    retained for reconstruction.  It does **not** cover the output array
    (``all_labels``) produced by :func:`_enumerate_from_parent_dag`, whose
    size is output-sensitive and can dominate on highly degenerate lattices
    (e.g. balanced kagome at (3,3) and above).
    Output size is bounded separately by the ``max_optima`` parameter on
    :meth:`FrontierExactSolver.solve`.

    Formula: ``2^3.9 × dp_cost^0.99 × n^1.2 × 1.5`` bytes, an empirical OLS fit
    with a safety multiplier.  This is a pre-solve estimate, not a conservative
    bound on every reconstruction solve.

    Args:
        dp_cost: Summed DP state cost ``Σ_steps 2^frontier_size`` for the
            chosen vertex ordering (see :func:`compute_dp_cost`).  At least 1.
        n: Number of vertices in the graph.
    """
    safety_multiplier = 1.5  # M: bare OLS fit underestimates ~half its sample.
    return int((2**3.9) * (dp_cost**0.99) * (n**1.2) * safety_multiplier)


def select_heuristic(graph: nx.Graph[int]) -> tuple[str, int]:
    """Select the best vertex-ordering heuristic by minimum frontier width.

    Evaluates ``"geometric"``, ``"ck"``, and ``"greedy"`` heuristics. Geometric
    already subsumes natural as one of its sweep candidates. Ties are broken in
    the order geometric > ck > greedy.

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
