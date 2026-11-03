"""Fast output tests for the typed solve workflow."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.fast


class _Graph:
    """Minimal graph value used by the workflow's cut calculations."""

    def __init__(self, edges: tuple[tuple[int, int], ...]) -> None:
        self._edges = edges

    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return the fixed unit-weight edges."""
        return self._edges

    def number_of_edges(self) -> int:
        """Return the number of fixed unit-weight edges."""
        return len(self._edges)


class _GraphTiling:
    """Deterministic tiling double exposing the workflow graph seam."""

    def __init__(self, graph: _Graph) -> None:
        self._graph = graph

    def graph(self, supercell: tuple[int, int], *, pbc: bool = True) -> _Graph:
        del supercell, pbc
        return self._graph


def _load_workflow(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load solve workflow source with only its lightweight dependency seams."""
    assignment = ModuleType("archimono.assignment")
    assignment.AnnealingSolver = object
    assignment.AssignmentResult = object
    assignment.AssignmentSolver = object
    assignment.BruteforceSolver = object
    assignment.FrontierExactSolver = object
    assignment.GreedySolver = object
    assignment.HadlockSolver = object
    assignment.compute_dp_cost = lambda graph, heuristic: 0.0
    assignment.estimate_reconstruct_memory_bytes = lambda dp_cost, n: 0
    assignment.SOLVERS_BY_NAME = {
        "bruteforce": assignment.BruteforceSolver,
        "frontier-dp": assignment.FrontierExactSolver,
        "annealing": assignment.AnnealingSolver,
        "greedy": assignment.GreedySolver,
        "hadlock": assignment.HadlockSolver,
    }
    tilings = ModuleType("archimono.tilings")
    tilings.registry = SimpleNamespace(get=lambda key: None)
    tilings.parse_supercell = lambda text: tuple(
        int(part) for part in text.replace("x", ",").split(",")
    )
    networkx = ModuleType("networkx")
    networkx.Graph = _Graph
    for name, module in {
        "archimono.assignment": assignment,
        "archimono.tilings": tilings,
        "networkx": networkx,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = (
        pathlib.Path(__file__).resolve().parents[3]
        / "src"
        / "archimono"
        / "workflows"
        / "solve.py"
    )
    module_name = "_fast_cli_solve_workflow"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _install_graph_tiling(
    monkeypatch: pytest.MonkeyPatch, workflow: ModuleType
) -> _Graph:
    """Install the fixed four-edge graph used by controlled solver results."""
    graph = _Graph(((0, 1), (1, 2), (2, 3), (3, 0)))
    monkeypatch.setattr(workflow.registry, "get", lambda key: _GraphTiling(graph))
    return graph


def _solve_with_missing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Graph, object]:
    """Run a fixed assignment result with its graph-derived metrics omitted."""
    workflow = _load_workflow(monkeypatch)
    graph = _install_graph_tiling(monkeypatch, workflow)
    result = SimpleNamespace(
        labels=(0, 1, 1, 0),
        objective_value=0.0,
        cut_value=None,
        n_frustrated=None,
        solver="controlled",
        metadata={"source": "test"},
    )
    solver = SimpleNamespace(solve=lambda graph, species, n_b: result)
    monkeypatch.setitem(workflow.SOLVERS_BY_NAME, "bruteforce", lambda: solver)
    solved = workflow.run_solve(workflow.SolveConfig(tiling="fixed"))
    return graph, solved


def test_assignment_without_solver_metrics_recomputes_cut_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assignment output derives its cut from graph edges when the solver omits it."""
    _, solved = _solve_with_missing_metrics(monkeypatch)

    assert solved.cut_value == 2


def test_assignment_without_solver_metrics_recomputes_frustrated_edge_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing frustrated-edge metrics equal total edges minus the derived cut."""
    graph, solved = _solve_with_missing_metrics(monkeypatch)

    assert solved.n_frustrated == graph.number_of_edges() - solved.cut_value


def test_solve_text_renders_custom_species_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _load_workflow(monkeypatch)
    result = workflow.SolveResult(
        tiling="fixed",
        supercell=(1, 1),
        pbc=True,
        solver="controlled",
        species=("Cu", "Zn"),
        labels=(0, 1, 0),
        cut_value=2,
        n_frustrated=1,
        metadata={},
    )

    assert "labels: Cu Zn Cu" in workflow.format_text(result)


def test_solve_json_keeps_assignment_labels_as_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _load_workflow(monkeypatch)
    result = workflow.SolveResult(
        tiling="fixed",
        supercell=(1, 1),
        pbc=True,
        solver="controlled",
        species=("Cu", "Zn"),
        labels=(0, 1),
        cut_value=1,
        n_frustrated=0,
        metadata={},
    )

    payload = json.loads(workflow.format_json(result))
    assert payload["labels"] == [0, 1]
    assert all(isinstance(label, int) for label in payload["labels"])
