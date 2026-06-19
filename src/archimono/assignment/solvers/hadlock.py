"""Unconstrained planar MAX-CUT via Hadlock's dual-graph T-join reduction.

Solves the unconstrained MAX-CUT problem on planar connected graphs in
polynomial time by reducing it to minimum-weight T-join / perfect matching
on the dual graph.

References
----------
Hadlock, F. (1975).
    Finding a Maximum Cut of a Planar Graph in Polynomial Time.
    *SIAM J. Comput.* 4(3), 221–225.
    https://doi.org/10.1137/0204019
    The T-join reduction (Eq. 3) and matching reduction (§3) used by
    ``_solve_unbalanced_hadlock``.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
from networkx.algorithms import matching, planarity

from archimono.assignment.solvers import _scoring, _validation, base

_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class _Face:
    """A face boundary in a planar embedding."""

    face_id: int
    half_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class _DualEdge:
    """One dual edge corresponding to one primal edge."""

    edge_id: int
    face_u: int
    face_v: int
    weight: float
    primal_edge: tuple[int, int]

    def other(self, face: int) -> int:
        """Return the other endpoint face of the dual edge."""
        if face == self.face_u:
            return self.face_v
        if face == self.face_v:
            return self.face_u
        raise ValueError(f"Face {face} is not incident to dual edge {self.edge_id}.")


class HadlockSolver(base.AssignmentSolver):
    """Unconstrained planar MAX-CUT solver via Hadlock's T-join reduction.

    Solves the unconstrained MAX-CUT problem on planar connected graphs in
    polynomial time.  The partition sizes are determined by the algorithm and
    cannot be constrained.  For fixed-composition MAX-CUT, use
    :class:`~archimono.assignment.solvers.bruteforce.BruteforceSolver` or
    :class:`~archimono.assignment.solvers.frontier_dp.FrontierExactSolver`.

    Requires:
        - The input graph must be planar.
        - The input graph must be connected.

    References:
        Hadlock (1975), *Finding a Maximum Cut of a Planar Graph in Polynomial
        Time*, SIAM J. Comput. 4(3), pp. 221–225.
        https://doi.org/10.1137/0204019
    """

    def _target_composition(
        self,
        n_nodes: int,
        n_b: int | None,
    ) -> base.TargetComposition | None:
        """Return ``None`` — Hadlock solves the unconstrained problem.

        Args:
            n_nodes: Total number of graph nodes.
            n_b: Must be ``None``. Passing a value raises ``ValueError``
                because the Hadlock algorithm does not support fixed-composition
                constraints.

        Returns:
            Always ``None``.

        Raises:
            ValueError: If ``n_b`` is not ``None``.
        """
        if n_b is not None:
            raise ValueError(
                "HadlockSolver solves the unconstrained planar MAX-CUT problem "
                "and does not accept a composition constraint (n_b). "
                "Use BruteforceSolver or FrontierExactSolver for fixed-composition "
                "MAX-CUT."
            )
        return None

    @staticmethod
    def _require_planar_embedding(
        graph: nx.Graph[int],
    ) -> planarity.PlanarEmbedding[int]:
        """Return a planar embedding or raise if the graph is non-planar.

        Args:
            graph: Assignment graph to embed.

        Returns:
            A planar embedding of ``graph``.

        Raises:
            ValueError: If ``graph`` is not planar.
        """
        is_planar, embedding = nx.check_planarity(graph)
        if not is_planar:
            raise ValueError(
                "HadlockSolver requires a planar graph. "
                "Use FrontierExactSolver or AnnealingSolver for non-planar graphs."
            )
        return planarity.PlanarEmbedding(embedding)

    def solve(
        self,
        graph: nx.Graph[int],
        species: Sequence[str],
        *,
        n_b: int | None = None,
    ) -> base.AssignmentResult:
        """Solve the unconstrained planar MAX-CUT problem exactly.

        Args:
            graph: Connected planar assignment graph whose nodes are labeled by
                integer indices.
            species: Ordered species labels, such as ``["A", "B"]``.
            n_b: Must be ``None``. Passing a value raises ``ValueError``.

        Returns:
            The exact assignment that maximizes the cut without any composition
            constraint.

        Raises:
            ValueError: If ``n_b`` is not ``None``, if the graph is not planar,
                if the graph is disconnected, or if the problem is not a valid
                binary species assignment.
            RuntimeError: If the dual reduction cannot reconstruct a valid cut.
        """
        _validation.validate_binary_species_problem(graph, species)
        self._target_composition(graph.number_of_nodes(), n_b)

        if not nx.is_connected(graph):
            raise ValueError(
                "HadlockSolver requires a connected planar graph for the "
                "Hadlock T-join reduction."
            )

        edges = _scoring.weighted_edge_list(graph)
        _validate_edge_weights(edges)

        embedding = self._require_planar_embedding(graph)

        labels, cut_value, metadata = _solve_unbalanced_hadlock(
            embedding=embedding,
            graph=graph,
        )

        n_frustrated = sum(
            1 for u, v in graph.edges() if labels[int(u)] == labels[int(v)]
        )
        return base.AssignmentResult(
            labels=labels,
            objective_value=cut_value,
            cut_value=cut_value,
            n_frustrated=n_frustrated,
            solver=self.__class__.__name__,
            metadata={
                "species": tuple(species),
                "label_mapping": {0: species[0], 1: species[1]},
                "target_composition": None,
                **metadata,
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _validate_edge_weights(edges: Sequence[base.AssignmentEdge]) -> None:
    """Validate the weighted edges accepted by the Hadlock solver.

    Args:
        edges: Weighted graph edges as ``(i, j, weight)`` triples.

    Raises:
        ValueError: If any edge weight is non-positive.
    """
    if any(weight <= 0.0 for _, _, weight in edges):
        raise ValueError("HadlockSolver requires strictly positive edge weights.")


def _solve_unbalanced_hadlock(
    *,
    embedding: planarity.PlanarEmbedding[int],
    graph: nx.Graph[int],
) -> tuple[base.AssignmentLabels, float, dict[str, object]]:
    """Solve unconstrained planar MAX-CUT via Hadlock's dual reduction.

    Args:
        embedding: Planar embedding of ``graph`` produced by NetworkX.
        graph: Connected planar assignment graph.

    Returns:
        A tuple containing the optimal assignment labels, the corresponding
        cut value, and solver metadata for the Hadlock reduction.

    Raises:
        RuntimeError: If the dual odd-face set cannot be matched or if the
            recovered primal cut is inconsistent.
    """
    faces, face_of_half_edge = _enumerate_faces(embedding)
    dual_edges, dual_adjacency = _build_dual_graph(
        face_of_half_edge=face_of_half_edge,
        graph=graph,
    )
    odd_faces = [face.face_id for face in faces if len(face.half_edges) % 2 == 1]

    if odd_faces:
        pairing_edges = _minimum_t_join_edges(
            dual_adjacency=dual_adjacency,
            dual_edges=dual_edges,
            odd_faces=odd_faces,
        )
        cover_edges = {dual_edges[edge_id].primal_edge for edge_id in pairing_edges}
    else:
        cover_edges = set()

    labels = _labels_from_cover(
        cover_edges=cover_edges,
        graph=graph,
    )
    cut_value = _scoring.maxcut_value(
        labels,
        _scoring.weighted_edge_list(graph),
    )
    cover_weight = sum(
        float(graph.edges[u, v].get("weight", 1.0)) for u, v in cover_edges
    )
    return labels, cut_value, {
        "algorithm": "hadlock_t_join",
        "n_faces": len(faces),
        "n_odd_faces": len(odd_faces),
        "odd_face_ids": tuple(odd_faces),
        "cover_weight": cover_weight,
    }


def _enumerate_faces(
    embedding: planarity.PlanarEmbedding[int],
) -> tuple[list[_Face], dict[tuple[int, int], int]]:
    """Enumerate all faces of a planar embedding.

    Args:
        embedding: Planar embedding whose half-edges define the face cycles.

    Returns:
        A list of face records and a mapping from each directed half-edge to
        the id of the face that contains it.
    """
    faces: list[_Face] = []
    face_of_half_edge: dict[tuple[int, int], int] = {}
    seen_half_edges: set[tuple[int, int]] = set()

    for start_u, start_v in embedding.edges():
        half_edge = (int(start_u), int(start_v))
        if half_edge in seen_half_edges:
            continue

        boundary: list[tuple[int, int]] = []
        current = half_edge
        while True:
            seen_half_edges.add(current)
            boundary.append(current)
            next_u, next_v = embedding.next_face_half_edge(*current)
            current = (int(next_u), int(next_v))
            if current == half_edge:
                break

        face_id = len(faces)
        for directed_edge in boundary:
            face_of_half_edge[directed_edge] = face_id
        faces.append(_Face(face_id=face_id, half_edges=tuple(boundary)))

    return faces, face_of_half_edge


def _build_dual_graph(
    *,
    face_of_half_edge: dict[tuple[int, int], int],
    graph: nx.Graph[int],
) -> tuple[list[_DualEdge], dict[int, list[int]]]:
    """Build the weighted dual multigraph induced by the primal graph.

    Args:
        face_of_half_edge: Mapping from directed primal half-edges to face ids.
        graph: Planar primal graph.

    Returns:
        The dual-edge records and a face-indexed adjacency list of dual-edge
        ids.
    """
    dual_edges: list[_DualEdge] = []
    dual_adjacency: dict[int, list[int]] = {}
    sorted_edges = sorted(
        graph.edges(data=True),
        key=lambda edge: (
            min(int(edge[0]), int(edge[1])),
            max(int(edge[0]), int(edge[1])),
        ),
    )

    for edge_id, (u, v, data) in enumerate(sorted_edges):
        primal_u = int(u)
        primal_v = int(v)
        primal_edge = (
            min(primal_u, primal_v),
            max(primal_u, primal_v),
        )
        dual_edge = _DualEdge(
            edge_id=edge_id,
            face_u=face_of_half_edge[(primal_u, primal_v)],
            face_v=face_of_half_edge[(primal_v, primal_u)],
            weight=float(data.get("weight", 1.0)),
            primal_edge=primal_edge,
        )
        dual_edges.append(dual_edge)
        dual_adjacency.setdefault(dual_edge.face_u, []).append(edge_id)
        if dual_edge.face_v != dual_edge.face_u:
            dual_adjacency.setdefault(dual_edge.face_v, []).append(edge_id)

    for adjacency in dual_adjacency.values():
        adjacency.sort()

    return dual_edges, dual_adjacency


def _minimum_t_join_edges(
    *,
    dual_adjacency: dict[int, list[int]],
    dual_edges: Sequence[_DualEdge],
    odd_faces: Sequence[int],
) -> set[int]:
    """Return the dual-edge ids of a minimum T-join on the odd faces.

    Args:
        dual_adjacency: Face-indexed dual adjacency as dual-edge ids.
        dual_edges: Weighted dual-edge records.
        odd_faces: Face ids with odd boundary length in the primal embedding.

    Returns:
        The set of dual-edge ids in the minimum T-join.

    Raises:
        RuntimeError: If the odd-face set cannot be perfectly matched in the
            dual metric closure.
    """
    if len(odd_faces) % 2 != 0:
        raise RuntimeError("The dual graph must contain an even number of odd faces.")

    distances: dict[int, dict[int, float]] = {}
    predecessors: dict[int, dict[int, tuple[int, int]]] = {}
    for source in odd_faces:
        dist_map, predecessor_map = _dual_dijkstra(
            dual_adjacency=dual_adjacency,
            dual_edges=dual_edges,
            source=source,
        )
        distances[source] = dist_map
        predecessors[source] = predecessor_map

    metric_closure: nx.Graph[int] = nx.Graph()
    metric_closure.add_nodes_from(int(face) for face in odd_faces)
    for index, source in enumerate(odd_faces):
        for target in odd_faces[index + 1 :]:
            distance = distances[source].get(target)
            if distance is None:
                raise RuntimeError(
                    "The dual graph is disconnected on the odd-face set."
                )
            metric_closure.add_edge(int(source), int(target), weight=distance)

    face_matching = matching.min_weight_matching(metric_closure, weight="weight")
    if len(face_matching) * 2 != len(odd_faces):
        raise RuntimeError("Failed to compute a perfect matching of the odd faces.")

    pairing_edges: set[int] = set()
    for source, target in face_matching:
        path_edges = _reconstruct_dual_path(
            predecessors=predecessors[int(source)],
            source=int(source),
            target=int(target),
        )
        for edge_id in path_edges:
            if edge_id in pairing_edges:
                pairing_edges.remove(edge_id)
            else:
                pairing_edges.add(edge_id)

    return pairing_edges


def _dual_dijkstra(
    *,
    dual_adjacency: dict[int, list[int]],
    dual_edges: Sequence[_DualEdge],
    source: int,
) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
    """Return shortest-path data from one source face in the dual graph.

    Args:
        dual_adjacency: Face-indexed dual adjacency as dual-edge ids.
        dual_edges: Weighted dual-edge records.
        source: Source face id.

    Returns:
        The shortest-path distances from ``source`` and predecessor data needed
        to reconstruct those paths.
    """
    distances = {source: 0.0}
    predecessors: dict[int, tuple[int, int]] = {}
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        distance, face = heapq.heappop(heap)
        if distance > distances[face] + _TOLERANCE:
            continue

        for edge_id in dual_adjacency.get(face, []):
            dual_edge = dual_edges[edge_id]
            other_face = dual_edge.other(face)
            if other_face == face:
                continue
            candidate_distance = distance + dual_edge.weight
            incumbent = distances.get(other_face, math.inf)
            if candidate_distance < incumbent - _TOLERANCE:
                distances[other_face] = candidate_distance
                predecessors[other_face] = (face, edge_id)
                heapq.heappush(heap, (candidate_distance, other_face))
                continue
            if math.isclose(candidate_distance, incumbent, abs_tol=_TOLERANCE):
                candidate = (face, edge_id)
                if candidate < predecessors.get(other_face, candidate):
                    predecessors[other_face] = candidate

    return distances, predecessors


def _reconstruct_dual_path(
    *,
    predecessors: dict[int, tuple[int, int]],
    source: int,
    target: int,
) -> list[int]:
    """Reconstruct one shortest dual path as dual-edge ids.

    Args:
        predecessors: Predecessor map produced by ``_dual_dijkstra``.
        source: Source face id.
        target: Target face id.

    Returns:
        The ordered dual-edge ids along the shortest path from ``source`` to
        ``target``.

    Raises:
        RuntimeError: If no predecessor chain reaches ``source``.
    """
    path_edge_ids: list[int] = []
    current = target
    while current != source:
        if current not in predecessors:
            raise RuntimeError(
                f"No dual path found from face {source} to face {target}."
            )
        previous, edge_id = predecessors[current]
        path_edge_ids.append(edge_id)
        current = previous
    path_edge_ids.reverse()
    return path_edge_ids


def _labels_from_cover(
    *,
    cover_edges: set[tuple[int, int]],
    graph: nx.Graph[int],
) -> base.AssignmentLabels:
    """Recover a node labeling whose cut is the complement of the cover.

    Args:
        cover_edges: Primal edges excluded from the cut by the dual solution.
        graph: Primal assignment graph.

    Returns:
        A canonical ``0/1`` labeling consistent with the recovered cut.

    Raises:
        RuntimeError: If the cover edges imply contradictory node labels.
    """
    labels: list[int | None] = [None] * graph.number_of_nodes()
    for start in range(graph.number_of_nodes()):
        if labels[start] is not None:
            continue
        labels[start] = 0
        queue: deque[int] = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in graph.neighbors(node):
                edge = (min(node, int(neighbor)), max(node, int(neighbor)))
                node_label = labels[node]
                if node_label is None:
                    raise RuntimeError("Node label reconstruction failed unexpectedly.")
                required = node_label if edge in cover_edges else 1 - node_label
                if labels[int(neighbor)] is None:
                    labels[int(neighbor)] = required
                    queue.append(int(neighbor))
                    continue
                if labels[int(neighbor)] != required:
                    raise RuntimeError(
                        "Recovered cover does not define a consistent cut."
                    )

    if any(label is None for label in labels):
        raise RuntimeError("Recovered cut did not assign every node.")

    labels_tuple = tuple(int(label) for label in labels)  # type: ignore[arg-type]
    flipped_tuple = tuple(1 - label for label in labels_tuple)
    return labels_tuple if labels_tuple <= flipped_tuple else flipped_tuple
