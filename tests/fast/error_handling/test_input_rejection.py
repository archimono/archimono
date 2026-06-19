"""Fast tests for input validation and error paths.

Every solver and enumerator must reject invalid inputs with clear errors.
"""

from __future__ import annotations

import networkx as nx
import pytest

from archimono.assignment import (
    AnnealingSolver,
    BruteforceSolver,
    GreedySolver,
    HadlockSolver,
)
from archimono.assignment.solvers.frontier_dp import FrontierExactSolver


class TestSolverRejectsInvalidInputs:
    """Solvers reject empty graphs and non-binary species."""

    @pytest.mark.parametrize(
        "solver_cls",
        [BruteforceSolver, AnnealingSolver, GreedySolver, FrontierExactSolver],
        ids=["bruteforce", "annealing", "greedy", "frontier_dp"],
    )
    def test_rejects_empty_graph(self, solver_cls: type) -> None:
        """Solver raises ValueError on empty graph."""
        graph: nx.Graph[int] = nx.Graph()
        with pytest.raises(ValueError, match="at least one node"):
            solver_cls().solve(graph, ["A", "B"])

    @pytest.mark.parametrize(
        "solver_cls",
        [BruteforceSolver, AnnealingSolver, GreedySolver, FrontierExactSolver],
        ids=["bruteforce", "annealing", "greedy", "frontier_dp"],
    )
    @pytest.mark.parametrize("species", [["A"], ["A", "B", "C"]])
    def test_rejects_non_binary_species(
        self, solver_cls: type, species: list[str]
    ) -> None:
        """Solver raises ValueError on non-binary species list."""
        graph = nx.path_graph(4)
        with pytest.raises(ValueError, match="two species"):
            solver_cls().solve(graph, species)

    def test_hadlock_rejects_non_planar(
        self, petersen_graph: nx.Graph[int]
    ) -> None:
        """HadlockSolver raises ValueError on non-planar graph."""
        with pytest.raises(ValueError, match="planar"):
            HadlockSolver().solve(petersen_graph, ["A", "B"])

    def test_hadlock_rejects_disconnected(self) -> None:
        """HadlockSolver raises ValueError on disconnected graph."""
        g: nx.Graph[int] = nx.Graph()
        g.add_edges_from([(0, 1), (2, 3)])
        with pytest.raises(ValueError, match="connected"):
            HadlockSolver().solve(g, ["A", "B"])

    def test_bruteforce_rejects_over_max_states(self) -> None:
        """BruteforceSolver raises ValueError when C(n, n_b) exceeds max_states."""
        g = nx.complete_graph(20)
        with pytest.raises(ValueError, match="exceeds"):
            BruteforceSolver(max_states=100).solve(g, ["A", "B"])

    def test_bruteforce_rejects_invalid_max_states(self) -> None:
        """BruteforceSolver rejects max_states < 1 at construction."""
        with pytest.raises(ValueError, match="max_states"):
            BruteforceSolver(max_states=0)

    def test_hadlock_rejects_non_positive_weight(self) -> None:
        """HadlockSolver raises ValueError on zero or negative edge weight."""
        g: nx.Graph[int] = nx.Graph()
        g.add_edge(0, 1, weight=0.0)
        g.add_edge(1, 2, weight=1.0)
        g.add_edge(2, 0, weight=1.0)
        with pytest.raises(ValueError, match="positive"):
            HadlockSolver().solve(g, ["A", "B"])

    def test_hadlock_rejects_nb(self, c4_graph: nx.Graph[int]) -> None:
        """HadlockSolver raises ValueError when n_b is passed."""
        with pytest.raises(ValueError, match="does not accept"):
            HadlockSolver().solve(c4_graph, ["A", "B"], n_b=2)


class TestSolverRejectsInvalidNb:
    """Solvers reject n_b values outside valid range."""

    @pytest.mark.parametrize(
        "solver_cls",
        [BruteforceSolver, AnnealingSolver, GreedySolver, FrontierExactSolver],
        ids=["bruteforce", "annealing", "greedy", "frontier_dp"],
    )
    @pytest.mark.parametrize("n_b", [-1, 5])
    def test_rejects_invalid_nb(self, solver_cls: type, n_b: int) -> None:
        """Every solver rejects n_b outside [0, n]."""
        graph = nx.path_graph(4)
        with pytest.raises(ValueError, match="n_b must satisfy"):
            solver_cls().solve(graph, ["A", "B"], n_b=n_b)


class TestAnnealingParameterValidation:
    """AnnealingSolver rejects invalid constructor parameters."""

    @pytest.mark.parametrize("temp", [0.0, -1.0])
    def test_rejects_invalid_temperature(self, temp: float) -> None:
        """Temperature must be positive."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            AnnealingSolver(temperature=temp)

    @pytest.mark.parametrize("rate", [0.0, 1.0, 1.5, -0.1])
    def test_rejects_invalid_cooling_rate(self, rate: float) -> None:
        """Cooling rate must be in (0, 1) exclusive."""
        with pytest.raises(
            ValueError, match="cooling_rate must lie strictly between 0 and 1"
        ):
            AnnealingSolver(cooling_rate=rate)
