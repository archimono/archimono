"""Base types and interfaces for assignment solvers."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from archimono.assignment.solvers import _validation

AssignmentLabel = int
AssignmentLabels = tuple[AssignmentLabel, ...]
AssignmentEdge = tuple[int, int, float]
TargetComposition = tuple[int, int]


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    """The output of a species-assignment solver.

    Attributes:
        labels: Ordered node labels encoded as `0` and `1`.
        objective_value: Weighted MAX-CUT objective value for the returned
            assignment.
        cut_value: Weighted MAX-CUT value associated with `labels`, if computed.
        n_frustrated: Number of non-cut edges, if computed.
        solver: Name of the solver that produced the result, if known.
        metadata: Extra solver-specific diagnostics and provenance information.
    """

    labels: AssignmentLabels
    objective_value: float
    cut_value: float | None = None
    n_frustrated: int | None = None
    solver: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the returned assignment labels."""
        _validation.validate_labels(self.labels)


class AssignmentSolver(abc.ABC):
    """Abstract interface for chemistry-agnostic species-assignment solvers."""

    @abc.abstractmethod
    def _target_composition(
        self,
        n_nodes: int,
        n_b: int | None,
    ) -> TargetComposition | None:
        """Return the solver's internal composition target.

        Args:
            n_nodes: Total number of graph nodes.
            n_b: Requested count of `1` labels, or `None` to request the
                solver's default behavior.

        Returns:
            A `(n_a, n_b)` pair for constrained solves, or `None` if the
            concrete solver interprets `n_b=None` as an unconstrained solve.

        Raises:
            ValueError: If `n_b` is invalid for the given graph size.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        graph: nx.Graph[int],
        species: Sequence[str],
        *,
        n_b: int | None = None,
    ) -> AssignmentResult:
        """Solve the assignment problem on a tiling graph.

        Args:
            graph: Undirected connectivity graph for the chosen tiling supercell.
            species: Ordered species labels, such as `["A", "B"]`.
            n_b: Optional count of `1` labels. If omitted, `None` is interpreted
                according to the concrete solver implementation.

        Returns:
            The best assignment that the solver finds for `graph`.
        """
        raise NotImplementedError
