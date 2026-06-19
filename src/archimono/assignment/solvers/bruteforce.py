"""Fixed-composition exact MAX-CUT by exhaustive enumeration.

Solves fixed-composition MAX-CUT by enumerating all ``C(n, n_b)`` binary
assignments and returning the one with maximum cut value.  This is exact
but exponential; use it only for small instances where the search space is
tractable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import networkx as nx

from archimono.assignment.solvers import _bruteforce, _scoring, _validation, base

_DEFAULT_MAX_STATES = 10_000_000


class BruteforceSolver(base.AssignmentSolver):
    """Exact fixed-composition MAX-CUT solver by exhaustive enumeration.

    Enumerates all ``C(n, n_b)`` binary assignments and returns the one
    with maximum weighted cut value.  Works on any graph topology — no
    planarity requirement.

    For unconstrained planar MAX-CUT, use
    :class:`~archimono.assignment.solvers.hadlock.HadlockSolver`.  For
    fixed-composition MAX-CUT on larger graphs, use
    :class:`~archimono.assignment.solvers.frontier_dp.FrontierExactSolver`.

    Args:
        max_states: Maximum number of candidate assignments to evaluate.
            Raises ``ValueError`` if ``C(n, n_b)`` exceeds this limit.
            Defaults to 10,000,000.
    """

    def __init__(self, max_states: int = _DEFAULT_MAX_STATES) -> None:
        if max_states < 1:
            raise ValueError("max_states must be at least 1.")
        self._max_states = max_states

    def _target_composition(
        self,
        n_nodes: int,
        n_b: int | None,
    ) -> base.TargetComposition:
        """Return the fixed composition target, defaulting ``n_b`` to ``n // 2``.

        Args:
            n_nodes: Total number of graph nodes.
            n_b: Requested count of ``1`` labels, or ``None`` to default to
                ``n_nodes // 2``.

        Returns:
            A ``(n_a, n_b)`` pair.

        Raises:
            ValueError: If ``n_b`` is not a valid integer in the range
                ``0 .. n_nodes``.
        """
        if n_b is None:
            n_b = n_nodes // 2
        if not isinstance(n_b, int) or isinstance(n_b, bool):
            raise ValueError("n_b must be an integer count of 1 labels.")
        if n_b < 0 or n_b > n_nodes:
            raise ValueError(f"n_b must satisfy 0 <= n_b <= {n_nodes}.")
        return (n_nodes - n_b, n_b)

    def solve(
        self,
        graph: nx.Graph[int],
        species: Sequence[str],
        *,
        n_b: int | None = None,
    ) -> base.AssignmentResult:
        """Solve fixed-composition MAX-CUT by exhaustive enumeration.

        Args:
            graph: Assignment graph whose nodes are labeled by integer indices.
            species: Ordered species labels, such as ``["A", "B"]``.
            n_b: Count of ``1`` labels.  Defaults to ``n // 2`` when ``None``.

        Returns:
            The exact assignment that maximizes the cut under the given
            composition constraint.

        Raises:
            TypeError: If ``species`` is passed as an integer instead of using
                the ``n_b`` keyword argument.
            ValueError: If the problem is not a valid binary species instance,
                if ``n_b`` is outside the valid range, or if ``C(n, n_b)``
                exceeds the ``max_states`` feasibility limit.
            RuntimeError: If no candidate assignments are evaluated.
        """
        _validation.validate_binary_species_problem(graph, species)

        target_composition = self._target_composition(
            graph.number_of_nodes(), n_b,
        )
        edges = _scoring.weighted_edge_list(graph)

        n_a, n_ones = target_composition
        n_total = n_a + n_ones
        n_states = math.comb(n_total, n_ones)
        if n_states > self._max_states:
            raise ValueError(
                f"Fixed-composition brute-force over C({n_total}, {n_ones}) = "
                f"{n_states:,} assignments exceeds the {self._max_states:,} "
                f"feasibility limit. Use AnnealingSolver or FrontierExactSolver "
                f"instead."
            )

        labels, cut_value, metadata = _bruteforce.solve_fixed_composition(
            edges=list(edges),
            graph=graph,
            target_composition=target_composition,
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
                "target_composition": target_composition,
                **metadata,
            },
        )
