"""Trivial 2-coloring assigner for bipartite tilings.

For a bipartite tiling the species-assignment problem is degenerate: the
graph admits a unique 2-coloring (up to global swap of the two species), and
that coloring is automatically optimal — every edge runs between the two
independent sets, so MAX-CUT equals ``|E|`` and there are zero frustrated
bonds.

All four bipartite Archimedean tilings (6³, 4⁴, 4.8², 4.6.12) are
**regular** bipartite graphs (every vertex has the same coordination number).
For a regular bipartite graph the two independent sets are necessarily
equal in size, so the only valid balanced stoichiometry is ``n_b == n / 2``.

This module exists so that the README pipeline diagram has a real
implementation behind its "yes — bipartite" branch, and so that downstream
code can call a single uniform interface regardless of whether the input
tiling is bipartite or frustrated.

References:
    NetworkX bipartite 2-coloring (BFS-based):
        Hagberg, A., Schult, D., & Swart, P. (2008).
        Exploring network structure, dynamics, and function using
        NetworkX.  *Proceedings of the 7th Python in Science
        Conference (SciPy 2008)*, 11–15.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from networkx.algorithms import bipartite as nx_bipartite

from archimono.assignment.solvers import _scoring, base

if TYPE_CHECKING:
    from archimono.tilings.base import Tiling


class BipartiteAssigner:
    """Compute the unique balanced 2-coloring of a bipartite tiling supercell.

    The assigner is intentionally not an :class:`AssignmentSolver` subclass:
    it operates on a *tiling* rather than a raw graph because it needs the
    tiling's declared bipartiteness as the source of truth, and because the
    natural caller-facing signature accepts a supercell shape directly.

    Example::

        >>> from archimono.tilings import registry
        >>> from archimono.assignment import BipartiteAssigner
        >>>
        >>> hexagonal = registry.get("hexagonal")
        >>> result    = BipartiteAssigner.assign(hexagonal, n_b=4, supercell=(2, 2))
        >>> result.cut_value          # all edges heterogeneous
        12.0
        >>> result.n_frustrated       # zero frustration
        0
        >>> len(result.labels)        # one label per site
        8
    """

    @staticmethod
    def assign(
        tiling: Tiling,
        n_b: int,
        supercell: tuple[int, int] = (1, 1),
    ) -> base.AssignmentResult:
        """Return the unique balanced 2-coloring of *tiling* over *supercell*.

        Args:
            tiling: A bipartite Archimedean tiling.  Non-bipartite
                tilings raise :class:`ValueError` — use a MAX-CUT
                solver from :mod:`archimono.assignment.solvers.annealing`
                instead.
            n_b: Number of label-1 atoms (sublattice B).  Must equal ``n / 2``
                where *n* is the number of sites in the supercell,
                since all bipartite Archimedean tilings are regular
                bipartite graphs whose two independent sets have
                equal size.
            supercell: ``(na, nb)`` supercell dimensions.

        Returns:
            An assignment with ``labels`` encoded as ``0/1`` for the
            two cut partitions (``0 =`` sublattice A,
            ``1 =`` sublattice B).  ``cut_value`` equals the total
            number of edges and ``n_frustrated`` is computed from
            the labeling (zero for any correctly bipartite
            supercell).

        Raises:
            ValueError: If *tiling* is not bipartite, if the
                requested supercell graph fails an independent
                NetworkX bipartiteness check (for example because of
                a parity artefact in an odd-sized supercell of a
                tiling that is bipartite only in the infinite limit),
                or if *n_b* does not equal ``n / 2``.
        """
        if not tiling.is_bipartite:
            raise ValueError(
                f"Tiling '{tiling.vertex_config}' is declared non-bipartite; "
                "use a MAX-CUT solver from archimono.assignment instead."
            )

        graph = tiling.graph(supercell)
        n = graph.number_of_nodes()

        # Defensive: Tiling.graph() now raises for invalid supercells before
        # reaching this point. This check is kept as a safety net in case the
        # graph was constructed outside of Tiling.graph().
        if not nx.is_bipartite(graph):
            raise ValueError(
                f"Tiling '{tiling.vertex_config}' is declared bipartite but the "
                f"{supercell[0]}×{supercell[1]} supercell graph is not bipartite. "
                "If this graph was built via Tiling.graph(), this should have been "
                "caught there — please report as a bug. Otherwise, use even "
                "values for both supercell dimensions."
            )

        if 2 * n_b != n:
            raise ValueError(
                f"n_b={n_b} is not balanced for a regular bipartite graph "
                f"with n={n} sites; expected n_b = {n // 2}."
            )

        # nx_bipartite.color works on disconnected components too, assigning
        # each component its own 0/1 labels via BFS. For all 11 Archimedean
        # tilings the supercell graph is connected, so the labeling is unique
        # up to a global swap.
        coloring: dict[int, int] = nx_bipartite.color(graph)
        n_ones = sum(1 for c in coloring.values() if c == 1)

        # Choose the orientation that puts exactly n_b sites at label 1.
        # label 1 = sublattice B; label 0 = sublattice A.
        if n_ones == n_b:
            labels = tuple(coloring[i] for i in range(n))
        else:
            labels = tuple(1 - coloring[i] for i in range(n))

        # Guardrail: a single global flip only balances a connected, regular
        # bipartite graph (the only kind any registry tiling produces). Reject
        # rather than silently return a labeling with the wrong composition.
        if sum(labels) != n_b:
            raise ValueError(
                f"BipartiteAssigner produced {sum(labels)} label-1 sites but "
                f"n_b={n_b}; the 2-coloring of this graph (e.g. a disconnected "
                f"or non-regular bipartite graph) cannot meet the requested "
                f"balanced composition."
            )

        edges = _scoring.weighted_edge_list(graph)
        cut_value = _scoring.maxcut_value(labels, edges)
        n_frustrated = sum(
            1 for u, v in graph.edges() if labels[int(u)] == labels[int(v)]
        )

        return base.AssignmentResult(
            labels=labels,
            objective_value=cut_value,
            cut_value=cut_value,
            n_frustrated=n_frustrated,
            solver="BipartiteAssigner",
        )
