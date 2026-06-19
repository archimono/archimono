"""Simulated annealing for binary assignment problems.

Implements a stochastic local-search solver for binary MAX-CUT under a
user-selected composition target. The solver preserves the number of `1`
and `0` labels by proposing swaps between opposite-label nodes and accepting
them with a Metropolis criterion under geometric cooling.

References
----------
Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. &
Teller, E. (1953).

    Equation of State Calculations by Fast Computing Machines.
    *J. Chem. Phys.* 21(6), 1087–1092.
    https://doi.org/10.1063/1.1699114
    The accept-with-exp(-ΔH/T) Metropolis criterion.

Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P. (1983).
    Optimization by Simulated Annealing.
    *Science* 220(4598), 671–680.
    https://doi.org/10.1126/science.220.4598.671
    Geometric cooling schedule and combinatorial-optimisation framing.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import networkx as nx

from archimono.assignment.solvers import (
    _heuristic_base,
    _scoring,
    _validation,
    base,
)


class AnnealingSolver(_heuristic_base.HeuristicAssignmentSolver):
    """Simulated annealing solver for composition-constrained binary MAX-CUT.

    Each restart begins from a random labeling with the required composition.
    The solver then proposes swaps between one `1`-labeled node and one
    `0`-labeled node, ensuring that the composition remains fixed
    throughout the search. If `n_b` is omitted, the solver defaults to a
    near-balanced target.
    """

    def __init__(
        self,
        *,
        temperature: float = 5.0,
        min_temperature: float = 1e-4,
        cooling_rate: float = 0.995,
        steps_per_temperature: int = 200,
        n_restarts: int = 50,
        seed: int | None = None,
    ) -> None:
        """Initialize the annealing solver.

        Args:
            temperature: Initial temperature for the Metropolis schedule.
            min_temperature: Final temperature cutoff for the schedule.
            cooling_rate: Multiplicative cooling factor applied each round.
            steps_per_temperature: Swap proposals per temperature level.
            n_restarts: Number of annealing restarts to evaluate.
            seed: Optional base seed used to derive restart-specific RNG states.

        Raises:
            ValueError: If any annealing hyperparameter is outside its valid
                range.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if min_temperature <= 0:
            raise ValueError("min_temperature must be positive.")
        if cooling_rate <= 0 or cooling_rate >= 1:
            raise ValueError("cooling_rate must lie strictly between 0 and 1.")
        if steps_per_temperature <= 0:
            raise ValueError("steps_per_temperature must be positive.")
        if n_restarts <= 0:
            raise ValueError("n_restarts must be positive.")

        self.temperature = temperature
        self.min_temperature = min_temperature
        self.cooling_rate = cooling_rate
        self.steps_per_temperature = steps_per_temperature
        self.n_restarts = n_restarts
        self.seed = seed

    def solve(
        self,
        graph: nx.Graph[int],
        species: Sequence[str],
        *,
        n_b: int | None = None,
    ) -> base.AssignmentResult:
        """Solve a binary assignment problem with simulated annealing.

        Args:
            graph: Assignment graph whose nodes are labeled by integer indices.
            species: Ordered species labels, such as `["A", "B"]`.
            n_b: Optional count of `1` labels. If omitted, initialize from the
                current near-balanced composition target.

        Returns:
            The best assignment found across all annealing restarts.

        Raises:
            ValueError: If the problem is not a non-empty two-species instance
                or if `n_b` lies outside the valid range.
            RuntimeError: If no assignment is produced.
        """

        _validation.validate_binary_species_problem(graph, species)

        edges = _scoring.weighted_edge_list(graph)
        target_composition = self._target_composition(graph.number_of_nodes(), n_b)
        adjacency = self._adjacency_weights(edges)
        best_result: base.AssignmentResult | None = None

        for restart in range(self.n_restarts):
            rng = random.Random(None if self.seed is None else self.seed + restart)
            result = self._anneal_once(
                graph=graph,
                edges=edges,
                adjacency=adjacency,
                rng=rng,
                species=species,
                restart=restart,
                target_composition=target_composition,
            )
            if (
                best_result is None
                or result.objective_value > best_result.objective_value
            ):
                best_result = result

        if best_result is None:
            raise RuntimeError("AnnealingSolver did not produce any assignment.")
        return best_result

    def _anneal_once(
        self,
        *,
        graph: nx.Graph[int],
        edges: list[base.AssignmentEdge],
        adjacency: _heuristic_base.AdjacencyMap,
        rng: random.Random,
        species: Sequence[str],
        restart: int,
        target_composition: base.TargetComposition,
    ) -> base.AssignmentResult:
        """Run one annealing restart and return its best assignment."""
        labels = self._initial_labels(target_composition)
        rng.shuffle(labels)
        one_nodes, zero_nodes = self._node_indices_by_label(labels)

        current_cut_value = _scoring.maxcut_value(labels, edges)
        best_labels = labels[:]
        best_cut_value = current_cut_value
        temperature = self.temperature

        while temperature > self.min_temperature and one_nodes and zero_nodes:
            for _ in range(self.steps_per_temperature):
                one_index = rng.randrange(len(one_nodes))
                zero_index = rng.randrange(len(zero_nodes))
                i = one_nodes[one_index]
                j = zero_nodes[zero_index]

                delta = self._swap_delta_cut_value(
                    i=i,
                    j=j,
                    labels=labels,
                    adjacency=adjacency,
                )

                if delta >= 0 or rng.random() < math.exp(delta / temperature):
                    labels[i] = 0
                    labels[j] = 1
                    self._swap_node_memberships(
                        one_nodes=one_nodes,
                        zero_nodes=zero_nodes,
                        one_index=one_index,
                        zero_index=zero_index,
                        i=i,
                        j=j,
                    )
                    current_cut_value += delta
                    if current_cut_value > best_cut_value:
                        best_cut_value = current_cut_value
                        best_labels = labels[:]

            temperature *= self.cooling_rate

        cut_value = _scoring.maxcut_value(best_labels, edges)
        return base.AssignmentResult(
            labels=tuple(best_labels),
            objective_value=best_cut_value,
            cut_value=cut_value,
            n_frustrated=sum(
                1
                for u, v in graph.edges()
                if best_labels[int(u)] == best_labels[int(v)]
            ),
            solver=self.__class__.__name__,
            metadata={
                "species": tuple(species),
                "label_mapping": {0: species[0], 1: species[1]},
                "target_composition": target_composition,
                "restart": restart,
                "n_restarts": self.n_restarts,
                "n_steps_per_temperature": self.steps_per_temperature,
            },
        )
